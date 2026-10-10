import json
from pathlib import Path

from pydantic import BaseModel

GOOD_TIERS = (1, 2)


class Question(BaseModel):
    id: str
    text: str
    search_queries: list[str] = []


class Source(BaseModel):
    id: str
    url: str
    title: str = ""
    domain: str = ""
    tier: int = 3
    weight: float = 0.4
    reasons: list[str] = []
    author: str = ""
    date: str = ""
    found_by: str = ""
    question_ids: list[str] = []
    abstract_only: bool = False
    text: str = ""


class Claim(BaseModel):
    id: str
    text: str
    quote: str = ""
    kind: str = "fact"
    question_id: str
    source_id: str


class ClaimGroup(BaseModel):
    id: str
    question_id: str
    main_claim: str
    claim_ids: list[str]
    supporting_sources: list[str] = []
    contradicting_sources: list[str] = []
    partial_sources: list[str] = []
    independent_sites: int = 0
    score: float = 0.0
    confidence: str = "UNVERIFIED"
    note: str = ""


class Section(BaseModel):
    heading: str
    paragraphs: list[str]


class KeyFinding(BaseModel):
    confidence: str
    text: str


class Report(BaseModel):
    title: str
    summary: list[KeyFinding] = []
    sections: list[Section] = []
    contested: list[str] = []
    limitations: list[str] = []


class ResearchState(BaseModel):
    topic: str
    depth: str
    started: str
    questions: list[Question] = []
    sources: list[Source] = []
    claims: list[Claim] = []
    groups: list[ClaimGroup] = []
    searches: list[str] = []
    failed_urls: list[str] = []
    extracted_sources: list[str] = []
    phases_done: list[str] = []
    report: Report | None = None

    def find_source(self, source_id):
        return next((s for s in self.sources if s.id == source_id), None)

    def source_for_url(self, url):
        return next((s for s in self.sources if s.url == url), None)

    def add_source(self, source):
        # the same page found for two questions is stored once and linked to both
        old = self.source_for_url(source.url)
        if old:
            for q in source.question_ids:
                if q not in old.question_ids:
                    old.question_ids.append(q)
            return old
        source.id = f"S{len(self.sources) + 1}"
        self.sources.append(source)
        return source

    def coverage(self):
        result = {}
        for q in self.questions:
            linked = [s for s in self.sources if q.id in s.question_ids]
            good = [s for s in linked if s.tier in GOOD_TIERS]
            sites = {s.domain for s in linked}
            result[q.id] = {"sources": len(linked), "good_sources": len(good), "websites": len(sites)}
        return result

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "state.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        return cls.model_validate(json.loads((Path(folder) / "state.json").read_text(encoding="utf-8")))
