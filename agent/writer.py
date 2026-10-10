import re

from pydantic import BaseModel

from agent import gemini
from agent.models import KeyFinding, Report, Section

USABLE = ("VERIFIED", "LIKELY")
CITATION = re.compile(r"\[(S\d+)\]")
CITATION_LIST = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)+)\]")

SYSTEM_PROMPT = (
    "You are the Writer in a research team. You write clear, neutral report text for a university reader. "
    "Use only the findings you are given and never add facts from your own knowledge. "
    "Every sentence must cite at least one source id in square brackets just before the full stop, for example: "
    "Global plastic production passed 400 million tonnes in 2022 [S3][S7]. "
    "State VERIFIED findings plainly. For LIKELY findings use careful wording such as 'evidence suggests'."
)


class SectionText(BaseModel):
    paragraphs: list[str]


class SummaryFinding(BaseModel):
    group_id: str
    sentence: str


class SummaryText(BaseModel):
    title: str
    key_findings: list[SummaryFinding]


def split_citation_lists(text):
    # turn [S1, S2] into [S1][S2] so every citation has the same format
    return CITATION_LIST.sub(lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))), text)


def split_sentences(paragraph):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", paragraph.strip()) if s]


def critic(paragraph, allowed_ids):
    # rule-based critic: a sentence is kept only if it cites at least one real source from the findings
    kept, removed = [], 0
    for sentence in split_sentences(split_citation_lists(paragraph)):
        cited = CITATION.findall(sentence)
        for sid in set(cited) - set(allowed_ids):
            sentence = sentence.replace(f"[{sid}]", "")
        if any(sid in allowed_ids for sid in cited):
            kept.append(re.sub(r"\s+([.!?,;])", r"\1", sentence).strip())
        else:
            removed += 1
    return " ".join(kept), removed


def finding_lines(groups):
    return "\n".join(f"- {g.id} [{g.confidence}] {g.main_claim} (sources: {' '.join(g.supporting_sources)})" for g in groups)


def write_section(state, question, emit):
    groups = [g for g in state.groups if g.question_id == question.id and g.confidence in USABLE]
    if not groups:
        return Section(heading=question.text, paragraphs=[
            "No finding for this question was supported by two independent websites or by one tier 1 source, "
            "so nothing is reported here. See the appendix for the claims that could not be checked."])
    prompt = (
        f"Research topic: {state.topic}\nSection question: {question.text}\n\nFindings:\n{finding_lines(groups)}\n\n"
        "Write a detailed answer to the section question in 2 to 4 paragraphs, using every finding above. "
        "Keep the numbers, dates, names and places from the findings, explain how the findings relate to each other, "
        "and cite the source ids listed for each finding."
    )
    result = gemini.ask_json(prompt, SectionText, system=SYSTEM_PROMPT)
    allowed = {sid for g in groups for sid in g.supporting_sources}
    paragraphs, removed = [], 0
    for p in result.paragraphs:
        text, dropped = critic(p, allowed)
        removed += dropped
        if text:
            paragraphs.append(text)
    emit("info", f"{question.id}: {len(paragraphs)} paragraphs written, critic removed {removed} uncited sentences")
    return Section(heading=question.text, paragraphs=paragraphs)


def write_summary(state):
    groups = [g for g in state.groups if g.confidence in USABLE]
    if not groups:
        return f"Research report: {state.topic}", []
    prompt = (
        f"Research topic: {state.topic}\n\nFindings:\n{finding_lines(groups)}\n\n"
        "Write a short report title and 4 to 8 key findings. Each key finding is one sentence based on one finding id, "
        "ending with that finding's source ids in square brackets. Prefer VERIFIED findings."
    )
    result = gemini.ask_json(prompt, SummaryText, system=SYSTEM_PROMPT)
    by_id = {g.id: g for g in groups}
    findings = []
    for item in result.key_findings:
        group = by_id.get(item.group_id)
        if group:
            text, _ = critic(item.sentence, set(group.supporting_sources))
            if text:
                findings.append(KeyFinding(confidence=group.confidence, text=text))
    return result.title.strip() or f"Research report: {state.topic}", findings


def contested_points(state):
    claims = {c.id: c for c in state.claims}
    points = []
    for g in state.groups:
        if g.confidence != "CONTESTED":
            continue
        support = "".join(f"[{s}]" for s in g.supporting_sources)
        disputed = [claims[cid] for cid in g.claim_ids if claims[cid].source_id in g.contradicting_sources]
        other = "; ".join(f'"{c.text}" [{c.source_id}]' for c in disputed[:3])
        note = f" {g.note}" if g.note else ""
        points.append(f"{g.main_claim.rstrip('. ')} {support}. Disputed by: {other}.{note}")
    return points


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def limitations(state):
    counts = {label: sum(1 for g in state.groups if g.confidence == label) for label in ("UNVERIFIED", "CONTESTED")}
    empty = [q.id for q in state.questions if not any(g.question_id == q.id and g.confidence in USABLE for g in state.groups)]
    notes = [
        "Trust tiers and confidence labels are estimates based on the type of website and on agreement between sources. "
        "They do not prove that a statement is true, so important facts should be checked in the cited sources.",
    ]
    if counts["UNVERIFIED"]:
        notes.append(f"{plural(counts['UNVERIFIED'], 'claim')} found in only one low-trust source "
                     "were left out of the findings (see the appendix).")
    if counts["CONTESTED"]:
        notes.append(f"{plural(counts['CONTESTED'], 'claim')} disputed by credible sources "
                     "are shown in the contested points section.")
    if empty:
        notes.append(f"No checked findings were available for: {', '.join(empty)}.")
    if state.failed_urls:
        notes.append(f"{plural(len(state.failed_urls), 'page')} could not be opened (paywalls, blocked downloads or pages without text).")
    abstract_only = sum(1 for s in state.sources if s.abstract_only)
    if abstract_only:
        notes.append(f"For {plural(abstract_only, 'scholarly source')} only the abstract was available, not the full paper.")
    notes.append("Sources are limited to what DuckDuckGo, Wikipedia and OpenAlex returned during this run, in English.")
    return notes


def write_report(state, emit):
    sections = [write_section(state, q, emit) for q in state.questions]
    title, summary = write_summary(state)
    emit("info", f"Summary written with {len(summary)} key findings")
    return Report(title=title, summary=summary, sections=sections, contested=contested_points(state), limitations=limitations(state))


def reference_numbers(state):
    # sources get numbers in the order they are first cited; uncited sources are numbered after them
    order = []
    texts = [f.text for f in state.report.summary] if state.report else []
    if state.report:
        texts += [p for s in state.report.sections for p in s.paragraphs] + state.report.contested
    for text in texts:
        for sid in CITATION.findall(text):
            if sid not in order:
                order.append(sid)
    order += [s.id for s in state.sources if s.id not in order]
    return {sid: i for i, sid in enumerate(order, start=1)}
