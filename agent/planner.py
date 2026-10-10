from pydantic import BaseModel

from agent import gemini
from agent.models import Question

SYSTEM_PROMPT = (
    "You are the Planner in a research team. You split a research topic into clear research questions "
    "that together cover the topic: background and definitions, key facts and numbers, causes and effects, "
    "current evidence, disagreements, and recent developments. Each question must be answerable from public sources."
)


class PlannedQuestion(BaseModel):
    question: str
    search_queries: list[str]


class Plan(BaseModel):
    questions: list[PlannedQuestion]


def make_plan(topic, how_many):
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Write exactly {how_many} research questions for this topic. "
        "For each question give 2 or 3 short web search queries (3 to 8 words each) that would find "
        "official, academic or reputable news sources. Do not number the questions."
    )
    plan = gemini.ask_json(prompt, Plan, system=SYSTEM_PROMPT)
    questions = []
    for i, item in enumerate(plan.questions[:how_many], start=1):
        questions.append(Question(id=f"Q{i}", text=item.question.strip(), search_queries=item.search_queries[:3]))
    if not questions:
        raise ValueError("The planner returned no research questions")
    return questions
