import re
from typing import Literal

import numpy as np
from pydantic import BaseModel

from agent import extractor, gemini
from agent.models import GOOD_TIERS, ClaimGroup
from agent.researcher import Researcher

SIMILARITY = 0.85
COPY_OVERLAP = 0.5
MAX_GROUP_SIZE = 10

SYSTEM_PROMPT = (
    "You are the Verifier in a research team. For each group of claims, the first claim is the main claim. "
    "Decide for every other claim whether it supports, contradicts or only partly supports the main claim. "
    "Pay close attention to numbers, dates, names and negations such as 'is' versus 'is not'. "
    "Judge only from the claim texts given to you, not from your own knowledge."
)


class Verdict(BaseModel):
    claim_id: str
    verdict: Literal["supports", "contradicts", "partial"]


class GroupCheck(BaseModel):
    group_id: str
    verdicts: list[Verdict]
    note: str


class CheckResult(BaseModel):
    groups: list[GroupCheck]


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def group_claims(claims, vectors, threshold=SIMILARITY):
    # leader grouping: a claim joins the first group whose first claim is similar enough, otherwise it starts a new group
    groups = []
    for claim, vector in zip(claims, vectors):
        for leader_vector, members in groups:
            if cosine(leader_vector, vector) >= threshold:
                members.append(claim)
                break
        else:
            groups.append((vector, [claim]))
    return [members for _, members in groups]


def shingles(text, size=8):
    words = re.findall(r"[a-z0-9]+", text.lower())[:4000]
    return {" ".join(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def find_copies(sources):
    # two sources sharing most 8-word phrases are copies of one story (e.g. a republished press release)
    phrase_sets = {s.id: shingles(s.text) for s in sources}
    owner = {s.id: s.id for s in sources}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            pa, pb = phrase_sets[a.id], phrase_sets[b.id]
            if pa and pb and len(pa & pb) / min(len(pa), len(pb)) >= COPY_OVERLAP:
                owner[b.id] = owner[a.id]
    return owner


def site_weights(source_ids, by_id, owner):
    # one vote per independent website: copies and pages from the same site count once, with their best weight
    best = {}
    for sid in source_ids:
        site = by_id[owner[sid]].domain
        best[site] = max(best.get(site, 0.0), by_id[sid].weight)
    return best


def decide_label(support_ids, contra_ids, by_id, owner):
    support = site_weights(support_ids, by_id, owner)
    contra = site_weights(contra_ids, by_id, owner)
    independent = len(support)
    best_tier = min((by_id[s].tier for s in support_ids), default=4)
    credible_contra = any(by_id[s].tier in GOOD_TIERS for s in contra_ids)
    score = round(sum(support.values()) - sum(contra.values()), 2)

    if credible_contra:
        label = "CONTESTED"
    elif independent >= 2 and best_tier in GOOD_TIERS:
        label = "VERIFIED"
    elif best_tier == 1 or independent >= 2:
        label = "LIKELY"
    else:
        label = "UNVERIFIED"
    return label, independent, score


def ask_for_verdicts(question, groups):
    lines = [f"Research question: {question.text}", ""]
    for local_id, members in groups:
        lines.append(f"Group {local_id}")
        lines.append(f"  main claim: {members[0].text}")
        for c in members[1:MAX_GROUP_SIZE]:
            lines.append(f"  {c.id} (from source {c.source_id}): {c.text}")
    lines.append("")
    lines.append("For every group give a verdict for each listed claim id and a one-sentence note describing "
                 "any disagreement (leave the note empty if all claims agree).")
    result = gemini.ask_json("\n".join(lines), CheckResult, system=SYSTEM_PROMPT)
    return {g.group_id: g for g in result.groups}


def check_question(state, question, owner):
    by_id = {s.id: s for s in state.sources}
    claims = [c for c in state.claims if c.question_id == question.id and c.kind != "opinion"]
    if not claims:
        return []
    # claims from the most trusted sources go first, so they become the main claim of their group
    claims.sort(key=lambda c: -by_id[c.source_id].weight)
    vectors = gemini.embed([c.text for c in claims])
    grouped = [(f"g{i}", members) for i, members in enumerate(group_claims(claims, vectors), start=1)]

    needs_check = [(gid, m) for gid, m in grouped if len({c.source_id for c in m}) >= 2]
    checks = ask_for_verdicts(question, needs_check) if needs_check else {}

    groups = []
    for gid, members in grouped:
        check = checks.get(gid)
        verdicts = {v.claim_id: v.verdict for v in check.verdicts} if check else {}
        support, contra, partial = {members[0].source_id}, set(), set()
        for c in members[1:]:
            verdict = verdicts.get(c.id, "supports")
            {"supports": support, "contradicts": contra, "partial": partial}[verdict].add(c.source_id)
        contra -= support
        partial -= support | contra
        label, independent, score = decide_label(sorted(support), sorted(contra), by_id, owner)
        groups.append(ClaimGroup(
            id="", question_id=question.id, main_claim=members[0].text, claim_ids=[c.id for c in members],
            supporting_sources=sorted(support), contradicting_sources=sorted(contra), partial_sources=sorted(partial),
            independent_sites=independent, score=score, confidence=label, note=check.note.strip() if check else "",
        ))
    return groups


def check_all(state, emit, question_ids=None):
    owner = find_copies(state.sources)
    keep = [g for g in state.groups if question_ids is not None and g.question_id not in question_ids]
    for q in state.questions:
        if question_ids is None or q.id in question_ids:
            groups = check_question(state, q, owner)
            keep += groups
            emit("info", f"{q.id}: {len(groups)} claim groups checked")
    order = {q.id: i for i, q in enumerate(state.questions)}
    keep.sort(key=lambda g: (order[g.question_id], -g.score))
    for i, g in enumerate(keep, start=1):
        g.id = f"G{i}"
    state.groups = keep


def recheck_weak_claims(state, settings, emit):
    # the verifier's loop: weak claims get a targeted search for better sources, then their question is checked again
    weak = sorted([g for g in state.groups if g.confidence == "UNVERIFIED"], key=lambda g: -g.score)[: settings["recheck"]]
    if not weak:
        return 0
    budget = dict(settings, max_sources=len(state.sources) + len(weak))
    helper = Researcher(state, budget, emit)
    new_sources = []
    for g in weak:
        query = " ".join(g.main_claim.split()[:12])
        emit("tool", f"targeted search for weak claim {g.id}: {query}")
        results = helper.run_search("web_search", {"query": query}).get("results", [])
        results += helper.run_search("paper_search", {"query": query}).get("results", [])
        candidates = sorted([r for r in results if not r.get("already_opened") and r["tier"] <= 2], key=lambda r: r["tier"])
        for r in candidates[:2]:
            opened = helper.open_source({"url": r["url"], "question_id": g.question_id})
            if "source_id" in opened:
                new_sources.append(state.find_source(opened["source_id"]))
                break
    if new_sources:
        extractor.extract_all(state, emit, sources=new_sources)
        check_all(state, emit, question_ids={g.question_id for g in weak})
    return len(new_sources)


def count_labels(groups):
    counts = {label: 0 for label in ("VERIFIED", "LIKELY", "CONTESTED", "UNVERIFIED")}
    for g in groups:
        counts[g.confidence] += 1
    return counts


def verify(state, settings, emit):
    check_all(state, emit)
    emit("info", "First check: " + ", ".join(f"{k.lower()} {v}" for k, v in count_labels(state.groups).items()))
    added = recheck_weak_claims(state, settings, emit)
    if added:
        emit("info", f"Targeted searches added {added} new sources")
    emit("info", "Final labels: " + ", ".join(f"{k.lower()} {v}" for k, v in count_labels(state.groups).items()))
