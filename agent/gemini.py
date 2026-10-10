import json
import random
import re
import time
from datetime import datetime, timedelta

import httpx
from google import genai
from google.genai import errors, types

from agent import cache, config

RETRY_CODES = {429, 500, 502, 503, 504}
BUSY_CODES = {503, 504}
MAX_TRIES = 6
BUSY_COOLDOWN = 300
QUOTA_WAIT = 3600

_client = None
# model -> time.time() when it may be used again (overloaded, or its daily free quota is used up)
_blocked_until = {}
_out_of_quota = set()


class GeminiUnavailable(RuntimeError):
    pass


def get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and paste your key.")
        # timeout is in milliseconds, so a stuck request fails instead of hanging for minutes
        options = types.HttpOptions(timeout=config.GEMINI_TIMEOUT * 1000)
        _client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=options)
    return _client


def models_to_try(model):
    # text models fall back along main -> backup -> lite; the embedding model has no fallback
    if model == config.EMBED_MODEL:
        return [model]
    chain = [model] + [m for m in (config.MAIN_MODEL, config.BACKUP_MODEL, config.LITE_MODEL) if m != model]
    if model == config.LITE_MODEL:
        chain = [config.LITE_MODEL, config.BACKUP_MODEL, config.MAIN_MODEL]
    return [m for i, m in enumerate(chain) if m and m not in chain[:i]]


def is_daily_quota(error):
    return getattr(error, "code", None) == 429 and "PerDay" in str(error)


def retry_seconds(error, default):
    found = re.search(r"'retryDelay': '(\d+)s'", str(error))
    return int(found.group(1)) if found else default


def when_free_again():
    soonest = min((_blocked_until[m] for m in _out_of_quota if m in _blocked_until), default=time.time() + QUOTA_WAIT)
    return (datetime.now() + timedelta(seconds=soonest - time.time())).strftime("%H:%M")


def with_retry(call, model):
    # call(model) sends the request; on busy errors or a used-up daily quota we move to the next model in the chain
    for current in models_to_try(model):
        if time.time() < _blocked_until.get(current, 0):
            continue
        for attempt in range(MAX_TRIES):
            try:
                return call(current)
            except (errors.APIError, httpx.TimeoutException) as e:
                code = getattr(e, "code", 504)
                if is_daily_quota(e):
                    _blocked_until[current] = time.time() + retry_seconds(e, QUOTA_WAIT)
                    _out_of_quota.add(current)
                    print(f"[gemini] daily free quota of {current} is used up, trying the next model")
                    break
                if code in BUSY_CODES and current != models_to_try(model)[-1]:
                    _blocked_until[current] = time.time() + BUSY_COOLDOWN
                    print(f"[gemini] {current} is busy (error {code}), skipping it for {BUSY_COOLDOWN // 60} minutes")
                    break
                if code not in RETRY_CODES or attempt == MAX_TRIES - 1:
                    raise
                # short limits (requests per minute) clear quickly, so wait 2s, 4s, 8s... and try again
                wait = min(60, 2 ** (attempt + 1)) + random.uniform(0, 1)
                print(f"[gemini] got error {code} from {current}, waiting {wait:.0f}s before retrying")
                time.sleep(wait)
    if any(m in _out_of_quota for m in models_to_try(model)):
        raise GeminiUnavailable(
            "The free daily Gemini quota is used up for every model this step can use. Google resets it about once a day; "
            f"the next model should be free again at about {when_free_again()}. Your progress is saved, use Resume then.")
    raise GeminiUnavailable("Every Gemini model is busy right now. Please try again in a few minutes.")


def ask(prompt, system=None, model=None, temperature=0.3):
    model = model or config.MAIN_MODEL
    key = json.dumps([model, system, prompt, temperature])
    cached = cache.load("gemini", key)
    if cached is not None:
        return cached

    settings = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    response = with_retry(lambda m: get_client().models.generate_content(model=m, contents=prompt, config=settings), model)
    text = response.text or ""
    cache.save("gemini", key, text)
    return text


def ask_json(prompt, schema, system=None, model=None, temperature=0.2):
    # schema is a pydantic class, Gemini is forced to reply with JSON that fits it
    model = model or config.MAIN_MODEL
    json_schema = schema.model_json_schema()
    key = json.dumps([model, system, prompt, temperature, json_schema])
    cached = cache.load("gemini", key)
    if cached is not None:
        return schema.model_validate_json(cached)

    settings = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=json_schema,
    )
    response = with_retry(lambda m: get_client().models.generate_content(model=m, contents=prompt, config=settings), model)
    result = schema.model_validate_json(response.text)
    cache.save("gemini", key, response.text)
    return result


def embed(texts, dimensions=768):
    vectors = [cache.load("embed", f"{config.EMBED_MODEL}|{dimensions}|{t}") for t in texts]
    missing = [t for t, v in zip(texts, vectors) if v is None]
    if missing:
        # each text must be its own Content, otherwise gemini-embedding-2 returns one vector for all of them
        contents = [types.Content(parts=[types.Part(text=t)]) for t in missing]
        settings = types.EmbedContentConfig(output_dimensionality=dimensions)
        response = with_retry(lambda m: get_client().models.embed_content(model=m, contents=contents, config=settings), config.EMBED_MODEL)
        if len(response.embeddings) != len(missing):
            raise ValueError(f"Asked for {len(missing)} embeddings but got {len(response.embeddings)}")
        new_vectors = iter([e.values for e in response.embeddings])
        for i, t in enumerate(texts):
            if vectors[i] is None:
                vectors[i] = next(new_vectors)
                cache.save("embed", f"{config.EMBED_MODEL}|{dimensions}|{t}", vectors[i])
    return vectors


def make_tools(tool_specs):
    # tool_specs: list of {"name", "description", "parameters"} where parameters is a JSON schema
    declarations = [
        types.FunctionDeclaration(name=s["name"], description=s["description"], parameters_json_schema=s["parameters"])
        for s in tool_specs
    ]
    return [types.Tool(function_declarations=declarations)]


def chat_step(history, tools, system=None, model=None):
    # one turn of the agent loop; our own code runs the tools, so the SDK must not call them automatically
    settings = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    return with_retry(lambda m: get_client().models.generate_content(model=m, contents=history, config=settings), model or config.MAIN_MODEL)


def user_message(text):
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def tool_results_message(results):
    # all answers of one turn go back in one message, with role "user" (the only role the API accepts) and marked as untrusted
    parts =[types.Part.from_function_response(name=name, response={"untrusted_data": result}) for name, result in results]
    return types.Content(role="user", parts=parts)


def tool_result_message(name, result):
    return tool_results_message([(name, result)])
