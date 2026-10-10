import html
import json
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agent import config, pipeline, writer
from agent.charts import TIER_COLORS, TIER_NAMES
from agent.models import ResearchState

st.set_page_config(page_title="DeepResearch Agent", page_icon=":material/travel_explore:", layout="wide")

ROOT = Path(__file__).resolve().parent
LABELS = ["VERIFIED", "LIKELY", "CONTESTED", "UNVERIFIED"]
BADGE_COLORS = {"VERIFIED": "green", "LIKELY": "blue", "CONTESTED": "orange", "UNVERIFIED": "red"}
LABEL_FILL = {"VERIFIED": "#B8FF9F", "LIKELY": "#A6FAFF", "CONTESTED": "#FFD23F", "UNVERIFIED": "#FF8A80"}
INK = "#111111"
LOG_LINES = 12
DEPTH_HELP = {
    "quick": "4 questions · up to 10 sources · about 20 Gemini calls",
    "standard": "6 questions · up to 20 sources · about 30 Gemini calls",
    "deep": "8 questions · up to 35 sources · about 45 Gemini calls",
}
EXAMPLE_TOPICS = [
    "Intermittent fasting vs. regular daily calorie restriction for weight loss: what does the clinical evidence say?",
    "Is the Great Wall of China visible from space with the naked eye?",
    "Solid-state batteries for electric vehicles: how close is mass production, and what are the main technical challenges?",
]
HERO = (
    '<div class="nb-hero"><span class="nb-sticker">AGENTIC AI · GEMINI</span>'
    '<h1>Deep<span class="nb-mark">Research</span> Agent</h1>'
    "<p>Give it a topic. The agent plans research questions, searches the web, news, Wikipedia and scholarly papers, "
    "cross-checks every fact against independent trusted sources and writes a detailed PDF report with citations.</p>"
    '<div class="nb-chips"><span>Gemini agent loop</span><span>Cross-validated facts</span>'
    "<span>Trust tiers</span><span>Cited PDF report</span></div></div>"
)

# the neo-brutalism look lives in style.css
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state():
    defaults = {"topic": "", "depth": config.DEPTH if config.DEPTH in config.DEPTH_SETTINGS else "standard",
                "running": False, "folder": None, "error": None, "example": None, "resume_folder": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def use_example():
    if st.session_state.example:
        st.session_state.topic = st.session_state.example
    st.session_state.example = None


def start_run():
    st.session_state.running = True
    st.session_state.error = None
    st.session_state.resume_folder = None


def resume_run(folder):
    st.session_state.running = True
    st.session_state.error = None
    st.session_state.resume_folder = folder


def open_earlier_run():
    st.session_state.folder = st.session_state.earlier_run
    st.session_state.error = None


def run_label(folder):
    stamp, _, slug = Path(folder).name.partition("_")
    when = datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%d %b, %H:%M")
    stopped = "" if pipeline.is_finished(folder) else " · stopped"
    return f"{when} · {slug.replace('-', ' ')}{stopped}"


def md(text):
    # Streamlit reads $...$ as maths, so dollar signs are escaped
    return text.replace("$", r"\$")


def cite(text, numbers, by_id):
    # [S3] becomes a clickable [3] that opens the original source
    for sid, n in numbers.items():
        if f"[{sid}]" in text:
            url = by_id[sid].url.replace("(", "%28").replace(")", "%29")
            text = text.replace(f"[{sid}]", f"[[{n}]]({url})")
    return md(text)


def refs(source_ids, numbers):
    return " ".join(f"[{numbers[s]}]" for s in source_ids if s in numbers)


def steps_html(active, finished=False):
    current = pipeline.PHASES.index(active) if active in pipeline.PHASES else -1
    chips = []
    for i, name in enumerate(pipeline.PHASES):
        css = "done" if finished or i < current else "active" if i == current else ""
        chips.append(f'<span class="nb-step {css}">{i + 1}. {name}</span>')
    return f'<div class="nb-steps">{"".join(chips)}</div>'


def log_html(events):
    # newest event first, so the latest agent action is always visible
    lines = "".join(
        f'<div class="nb-line"><span class="nb-tag {e["kind"]}">{e["kind"].upper()}</span>'
        f'<span class="nb-time">{e["time"]}</span>{html.escape(e["text"][:170]).replace("$", "&#36;")}</div>'
        for e in reversed(events[-LOG_LINES:]))
    return f'<div class="nb-log"><div class="nb-log-title">Live agent log · newest first</div>{lines}</div>'


def sidebar():
    with st.sidebar:
        st.markdown("### :material/travel_explore: DeepResearch Agent")
        st.caption("An agentic AI that researches a topic, cross-validates every fact and writes a cited PDF report.")
        st.segmented_control("Research depth", list(config.DEPTH_SETTINGS), key="depth", disabled=st.session_state.running)
        st.caption(DEPTH_HELP.get(st.session_state.depth or "standard", ""))
        st.divider()
        st.markdown("**Gemini models**")
        if config.GEMINI_API_KEY:
            st.badge("API key loaded", icon=":material/key:", color="green")
        else:
            st.badge("No API key in .env", icon=":material/key_off:", color="red")
        st.caption(f"Main: {config.MAIN_MODEL}  \nBackup: {config.BACKUP_MODEL}  \n"
                   f"Bulk work: {config.LITE_MODEL}  \nEmbeddings: {config.EMBED_MODEL}")
        st.caption("The free tier allows only a small number of requests per model each day. "
                   "If a run stops because the quota is used up, open it below and resume it later.")
        st.divider()
        runs = [str(f) for f in pipeline.list_runs()]
        if runs:
            st.selectbox("Open an earlier run", runs, index=None, key="earlier_run", format_func=run_label,
                         on_change=open_earlier_run, placeholder="Choose a run", disabled=st.session_state.running)
        else:
            st.caption("Finished reports will be listed here.")


def topic_form():
    st.markdown(HERO, unsafe_allow_html=True)
    with st.container(key="topic_box"):
        st.text_area("Research topic", key="topic", height=90, disabled=st.session_state.running,
                     placeholder="For example: How does India's UPI payment system work, and how fast has it grown?")
        st.pills("Try an example", EXAMPLE_TOPICS, key="example", on_change=use_example, disabled=st.session_state.running)
        ready = bool(st.session_state.topic.strip()) and bool(config.GEMINI_API_KEY)
        st.button("Start research", type="primary", icon=":material/search:", on_click=start_run,
                  disabled=st.session_state.running or not ready)


def run_research():
    resume_folder = st.session_state.resume_folder
    if resume_folder:
        topic = ResearchState.load(resume_folder).topic
    else:
        topic, depth = st.session_state.topic.strip(), st.session_state.depth or "standard"
    st.markdown(f"### Researching: {md(topic)}")
    steps = st.empty()
    with st.container(key="live_numbers"):
        numbers_box = st.empty()
    log_box = st.empty()
    events = []
    steps.markdown(steps_html("Plan"), unsafe_allow_html=True)

    def on_event(event, state):
        events.append(event)
        if event["kind"] == "phase":
            steps.markdown(steps_html(event["text"]), unsafe_allow_html=True)
        log_box.markdown(log_html(events), unsafe_allow_html=True)
        with numbers_box.container():
            cols = st.columns(4)
            cols[0].metric("Research questions", len(state.questions))
            cols[1].metric("Sources read", len(state.sources))
            cols[2].metric("Claims extracted", len(state.claims))
            cols[3].metric("Verified groups", sum(1 for g in state.groups if g.confidence == "VERIFIED"))

    try:
        if resume_folder:
            state, folder = pipeline.resume(resume_folder, on_event)
        else:
            state, folder = pipeline.run(topic, depth, on_event)
        steps.markdown(steps_html("Publish", finished=True), unsafe_allow_html=True)
        st.session_state.folder = str(folder)
    except Exception as e:
        st.session_state.error = str(e) if isinstance(e, pipeline.verifier.gemini.GeminiUnavailable) else f"{type(e).__name__}: {e}"
        runs = pipeline.list_runs()
        st.session_state.folder = resume_folder or (str(runs[0]) if runs else None)
    finally:
        st.session_state.running = False
        st.session_state.resume_folder = None
    st.rerun()


def bar(data, field, color_scale, title):
    # flat colours with thick black outlines, to match the neo-brutalism page style
    base = alt.Chart(data, title=alt.Title(title, anchor="start", font="Archivo Black", fontSize=15, fontWeight="normal", color=INK)).encode(
        y=alt.Y(f"{field}:N", sort=None, title=None, axis=alt.Axis(labelFont="Space Grotesk", labelFontSize=13, labelFontWeight="bold",
                                                                  labelColor=INK, domainColor=INK, domainWidth=3, ticks=False)),
        x=alt.X("count:Q", title=None, axis=None, scale=alt.Scale(domain=[0, max(1, int(data["count"].max())) * 1.15])),
        tooltip=[alt.Tooltip(c, title=c.title()) for c in data.columns],
    )
    bars = base.mark_bar(stroke=INK, strokeWidth=2.5, height={"band": 0.68}).encode(
        color=alt.Color(f"{field}:N", scale=color_scale, legend=None))
    labels = base.mark_text(align="left", dx=7, font="Space Grotesk", fontSize=14, fontWeight="bold", color=INK).encode(text="count:Q")
    return (bars + labels).properties(height=190, background="#FFFFFF").configure_view(stroke=None)


def show_summary(state, numbers, by_id):
    if not state.report.summary:
        st.info("No finding passed the cross-validation rules. The Claims tab shows what was found.")
    for finding in state.report.summary:
        color = BADGE_COLORS[finding.confidence]
        st.markdown(f":{color}-badge[{finding.confidence}] {cite(finding.text, numbers, by_id)}")


def show_findings(state, numbers, by_id):
    for i, section in enumerate(state.report.sections, start=1):
        st.subheader(f"{i}. {md(section.heading)}")
        for p in section.paragraphs:
            st.markdown(cite(p, numbers, by_id))


def show_sources(state, numbers):
    tiers = pd.DataFrame({"tier": [f"Tier {t}" for t in (1, 2, 3, 4)], "meaning": [TIER_NAMES[t][8:] for t in (1, 2, 3, 4)],
                          "count": [sum(1 for s in state.sources if s.tier == t) for t in (1, 2, 3, 4)]})
    labels = pd.DataFrame({"label": [label.title() for label in LABELS],
                           "count": [sum(1 for g in state.groups if g.confidence == label) for label in LABELS]})
    tier_scale = alt.Scale(domain=list(tiers["tier"]), range=[TIER_COLORS[t] for t in (1, 2, 3, 4)])
    label_scale = alt.Scale(domain=list(labels["label"]), range=[LABEL_FILL[label] for label in LABELS])
    left, right = st.columns(2)
    left.altair_chart(bar(tiers, "tier", tier_scale, "Sources read, by trust tier"), theme=None)
    right.altair_chart(bar(labels, "label", label_scale, "Claim groups, by confidence label"), theme=None)
    rows = [{"Ref": numbers[s.id], "Title": s.title or s.url, "Website": s.domain, "Tier": s.tier, "Weight": s.weight,
             "Found by": s.found_by, "Read": "abstract" if s.abstract_only else "full text", "Link": s.url,
             "Why this tier": "; ".join(s.reasons)} for s in state.sources]
    table = pd.DataFrame(rows).sort_values("Ref") if rows else pd.DataFrame()
    st.dataframe(table, hide_index=True, column_config={
        "Link": st.column_config.LinkColumn("Link", display_text="open"),
        "Weight": st.column_config.ProgressColumn("Weight", min_value=0.0, max_value=1.0, format="%.2f"),
    })


def show_claims(state, numbers):
    chosen = st.pills("Show labels", LABELS, default=LABELS, selection_mode="multi", key="label_filter")
    rows = [{"Group": g.id, "Question": g.question_id, "Label": g.confidence, "Claim": g.main_claim,
             "Supported by": refs(g.supporting_sources, numbers), "Disputed by": refs(g.contradicting_sources, numbers),
             "Websites": g.independent_sites, "Note": g.note} for g in state.groups if g.confidence in (chosen or [])]
    st.caption(f"{len(rows)} of {len(state.groups)} claim groups. Each group is a set of matching claims from different sources.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_method(state, folder):
    for q in state.questions:
        st.markdown(f"**{q.id}** {md(q.text)}")
    st.caption(f"{len(state.searches)} searches · {len(state.sources)} sources read · {len(state.failed_urls)} pages could not be opened")
    trace_file = Path(folder) / "trace.jsonl"
    if trace_file.exists():
        with st.expander("Agent trace (every step the agents took)"):
            events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            st.dataframe(pd.DataFrame(events), hide_index=True)
    with st.expander("Searches run"):
        st.dataframe(pd.DataFrame({"search": state.searches}), hide_index=True)
    if state.failed_urls:
        with st.expander("Pages that could not be opened"):
            for url in state.failed_urls:
                st.markdown(f"- {url}")
    st.caption(f"Run folder: `{folder}`")


def show_results(folder):
    folder = Path(folder)
    state = ResearchState.load(folder)
    if not pipeline.is_finished(folder) or state.report is None:
        with st.container(key="stopped_box"):
            st.markdown(f"#### Stopped run: {md(state.topic)}")
            done = ", ".join(state.phases_done) or "nothing yet"
            st.caption(f"Finished phases: {done} · {len(state.sources)} sources and {len(state.claims)} claims saved")
            reason = pipeline.last_error(folder)
            if reason:
                st.caption(f"Stopped because: {md(reason[:300])}")
            st.button("Resume this run", icon=":material/play_arrow:", type="primary", on_click=resume_run, args=(str(folder),),
                      disabled=st.session_state.running or not config.GEMINI_API_KEY)
        return
    numbers = writer.reference_numbers(state)
    by_id = {s.id: s for s in state.sources}
    pdf_file = folder / "report.pdf"

    with st.container(key="report_head"):
        head, action = st.columns([3, 1], vertical_alignment="center")
        head.markdown(f"## {md(state.report.title)}")
        head.caption(f"Topic: {md(state.topic)} · depth: {state.depth} · {run_label(folder).split(' · ')[0]}")
        if pdf_file.exists():
            action.download_button("Download PDF", pdf_file.read_bytes(), file_name=f"{folder.name}.pdf", mime="application/pdf",
                                   icon=":material/download:", type="primary", on_click="ignore")
        counts = " ".join(f":{BADGE_COLORS[k]}-badge[{k.title()} {sum(1 for g in state.groups if g.confidence == k)}]" for k in LABELS)
        st.markdown(counts)

    tabs = st.tabs(["Summary", "Findings", "Contested points", "Sources", "Claims", "Method", "PDF report"])
    with tabs[0]:
        show_summary(state, numbers, by_id)
    with tabs[1]:
        show_findings(state, numbers, by_id)
    with tabs[2]:
        if state.report.contested:
            for point in state.report.contested:
                st.markdown(f"- {cite(point, numbers, by_id)}")
        else:
            st.info("No credible source disagreed with a finding in this run.")
    with tabs[3]:
        show_sources(state, numbers)
    with tabs[4]:
        show_claims(state, numbers)
    with tabs[5]:
        show_method(state, folder)
    with tabs[6]:
        if pdf_file.exists():
            with st.container(key="pdf_view"):
                st.pdf(str(pdf_file), height=900)


init_state()
sidebar()
topic_form()
if st.session_state.error:
    st.error(st.session_state.error, icon=":material/error:")
if st.session_state.running:
    run_research()
elif st.session_state.folder:
    show_results(st.session_state.folder)
