import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from agent import config, extractor, pdf_report, planner, researcher, verifier, writer
from agent.models import ResearchState

PHASES = ["Plan", "Research", "Extract", "Cross-validate", "Write", "Publish"]


def make_run_folder(topic):
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "topic"
    folder = config.RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def copy_name(state):
    # readable file name for the copy in the main folder, e.g. "Microplastics in drinking water (2026-10-10 2028).pdf"
    safe = re.sub(r'[<>:"/\\|?*]+', " ", state.topic)
    safe = re.sub(r"\s+", " ", safe).strip()
    if len(safe) > 120:
        safe = safe[:120].rsplit(" ", 1)[0]
    safe = safe.rstrip(" .,") or "Research report"
    return f"{safe[0].upper()}{safe[1:]} ({datetime.now():%Y-%m-%d %H%M}).pdf"


def run(topic, depth="standard", on_event=None):
    topic = topic.strip()
    if not topic:
        raise ValueError("Please enter a research topic")
    if depth not in config.DEPTH_SETTINGS:
        raise ValueError(f"Unknown depth '{depth}', use one of {list(config.DEPTH_SETTINGS)}")
    folder = make_run_folder(topic)
    state = ResearchState(topic=topic, depth=depth, started=datetime.now().isoformat(timespec="seconds"))
    return run_phases(state, folder, on_event)


def resume(folder, on_event=None):
    # continue a stopped run from the first phase that did not finish
    folder = Path(folder)
    return run_phases(ResearchState.load(folder), folder, on_event)


def run_phases(state, folder, on_event):
    settings = config.DEPTH_SETTINGS[state.depth]
    current = {"phase": ""}

    with open(folder / "trace.jsonl", "a", encoding="utf-8") as trace:
        def emit(kind, text):
            # every event goes to trace.jsonl and to the screen (command line or web page)
            event = {"time": f"{datetime.now():%H:%M:%S}", "phase": current["phase"], "kind": kind, "text": text}
            trace.write(json.dumps(event, ensure_ascii=False) + "\n")
            trace.flush()
            if on_event:
                on_event(event, state)

        def plan():
            state.questions = planner.make_plan(state.topic, settings["questions"])
            for q in state.questions:
                emit("info", f"{q.id}: {q.text}")

        def research():
            researcher.run_research(state, settings, emit)
            if not state.sources:
                raise RuntimeError("The Researcher did not save any source, so there is nothing to check. "
                                   "Resume the run to try again, or try a different topic.")

        def extract():
            extractor.extract_all(state, emit)

        def cross_validate():
            verifier.verify(state, settings, emit)

        def write():
            state.report = writer.write_report(state, emit)

        def publish():
            pdf_path = pdf_report.build_pdf(state, folder / "report.pdf")
            config.REPORTS_COPY_DIR.mkdir(parents=True, exist_ok=True)
            copy = shutil.copy2(pdf_path, config.REPORTS_COPY_DIR / copy_name(state))
            emit("info", f"PDF also saved as {copy}")

        steps = dict(zip(PHASES, [plan, research, extract, cross_validate, write, publish]))
        try:
            if state.phases_done:
                emit("info", f"Resuming after: {', '.join(state.phases_done)}")
            for name, step in steps.items():
                if name in state.phases_done:
                    continue
                current["phase"] = name
                emit("phase", name)
                step()
                state.phases_done.append(name)
                state.save(folder)
            emit("done", str(folder / "report.pdf"))
            return state, folder
        except Exception as e:
            emit("error", f"{type(e).__name__}: {e}")
            state.save(folder)
            raise


def list_runs():
    # newest first; a run counts once it has saved its state
    if not config.RUNS_DIR.exists():
        return []
    return sorted((f for f in config.RUNS_DIR.iterdir() if (f / "state.json").exists()), reverse=True)


def is_finished(folder):
    return (Path(folder) / "report.pdf").exists()


def last_error(folder):
    trace_file = Path(folder) / "trace.jsonl"
    if not trace_file.exists():
        return ""
    errors = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if '"kind": "error"' in line]
    return errors[-1]["text"] if errors else ""
