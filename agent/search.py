import time

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from agent import cache, config

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
OPENALEX_API = "https://api.openalex.org/works"


def _get_json(url, params=None):
    response = requests.get(url, params=params, headers={"User-Agent": config.user_agent()}, timeout=config.REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _cached(kind, key, run_search):
    results = cache.load("search", f"{kind}|{key}")
    if results is None:
        results = run_search()
        # empty results are not saved, they are often a temporary block from the search engine
        if results:
            cache.save("search", f"{kind}|{key}", results)
    return results


def _duckduckgo(kind, query, max_results):
    # DuckDuckGo sometimes refuses a request for a moment, so try once more after a short pause
    for attempt in range(2):
        try:
            engine = DDGS()
            return engine.text(query, max_results=max_results) if kind == "web" else engine.news(query, max_results=max_results)
        except DDGSException as e:
            if attempt == 1:
                print(f"[search] DuckDuckGo {kind} search failed for '{query}': {e}")
                return []
            time.sleep(2)


def web_search(query, max_results=8):
    def run():
        hits = _duckduckgo("web", query, max_results)
        return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", ""), "found_by": "web"} for h in hits]

    return _cached("web", f"{query}|{max_results}", run)


def news_search(query, max_results=6):
    def run():
        hits = _duckduckgo("news", query, max_results)
        return [
            {"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("body", ""), "date": h.get("date", ""),
             "publisher": h.get("source", ""), "found_by": "news"}
            for h in hits
        ]

    return _cached("news", f"{query}|{max_results}", run)


def wikipedia_search(topic, max_results=3):
    def run():
        found = _get_json(WIKI_API, {"action": "query", "list": "search", "srsearch": topic, "srlimit": max_results, "format": "json"})
        results = []
        for item in found["query"]["search"]:
            page = _get_json(WIKI_SUMMARY + requests.utils.quote(item["title"].replace(" ", "_"), safe=""))
            results.append({
                "title": page.get("title", item["title"]),
                "url": page.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "snippet": page.get("extract", ""),
                "found_by": "wikipedia",
            })
        return results

    return _cached("wikipedia", f"{topic}|{max_results}", run)


def rebuild_abstract(inverted_index):
    # OpenAlex stores abstracts as {word: [positions]}, so put the words back in order
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


def paper_search(query, max_results=5):
    def run():
        params = {
            "search": query,
            "per_page": max_results,
            "select": "id,doi,display_name,publication_year,type,primary_location,best_oa_location,cited_by_count,abstract_inverted_index,is_retracted",
        }
        if config.CONTACT_EMAIL:
            params["mailto"] = config.CONTACT_EMAIL
        works = _get_json(OPENALEX_API, params)["results"]
        results = []
        for w in works:
            if w.get("is_retracted"):
                continue
            location = w.get("primary_location") or {}
            journal = (location.get("source") or {}).get("display_name", "")
            results.append({
                "title": w.get("display_name") or "",
                # the publisher's own page shows the real website (e.g. who.int), a doi.org link hides it
                "url": location.get("landing_page_url") or w.get("doi") or w.get("id", ""),
                "doi": w.get("doi") or "",
                "snippet": rebuild_abstract(w.get("abstract_inverted_index"))[:800],
                "date": str(w.get("publication_year") or ""),
                "publisher": journal,
                "work_type": w.get("type", ""),
                "cited_by": w.get("cited_by_count", 0),
                "pdf_url": (w.get("best_oa_location") or {}).get("pdf_url") or "",
                "found_by": "openalex",
            })
        return results

    return _cached("openalex", f"{query}|{max_results}", run)
