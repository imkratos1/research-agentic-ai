# Project plan: DeepResearch Agent

## Goal
Build an **agentic AI** for the AI course. An agentic AI is an LLM that chooses its own actions, uses tools, looks at the results, and keeps going until the goal is met.

Given any topic, the agent will:
1. Search the web in depth.
2. Extract facts from many sources.
3. Cross-validate each fact against independent, trusted sources.
4. Produce a detailed PDF report with citations.

**Fixed decisions:**
- The LLM is Google Gemini on the free tier.
- The agent loop is written from scratch in Python, with no LangChain or other agent framework.
- The project includes a Streamlit UI, a project report, and an evaluation.

The diagrams for every part of this plan are in [`diagrams/`](diagrams/) and are shown in the [README](README.md).

## One research run

A **pipeline** runs six phases in order. Phases 2 and 4 are **agent loops**, where Gemini picks the next tool and decides when to stop.

| # | Phase | Agent role | What happens |
|---|---|---|---|
| 1 | Plan | Planner | Split the topic into 4–8 research questions, each with 2–3 search queries |
| 2 | Research (loop) | Researcher | ReAct loop (Reason → Act → Observe): search the web, news, Wikipedia and OpenAlex; open the most useful sources; score their trust tier. Stop when every question has enough good sources or the step budget runs out |
| 3 | Extract | Extractor | Turn each source into single-fact claims: claim, supporting quote, source, and type (fact, statistic or opinion) |
| 4 | Cross-validate (loop) | Verifier | Group matching claims, check agreement, count independent websites, give a confidence label, and run targeted searches for weak claims |
| 5 | Write | Writer + Critic | Write each section and the summary using only Verified and Likely claims with source citations. The rule-based critic removes any sentence without a valid source |
| 6 | Publish | PDF builder | Build the PDF report with charts and references, and copy it into the main project folder |

Progress is saved after every phase, so a run that stops (for example when the daily quota is used up) can be resumed without repeating finished phases.

Each agent role is the same Gemini model with a different prompt and set of tools. This is a simple **multi-agent system**.

## AI concepts covered (for the project report)

| Concept | Simple explanation | Where in the code |
|---|---|---|
| ReAct agent loop | The model reasons, calls a tool, reads the result, and repeats | `researcher.py`, `verifier.py` |
| Function calling | The model replies with "call this tool with these arguments" instead of plain text | `gemini.py` (`make_tools`, `chat_step`) |
| Task decomposition | A big topic is split into smaller research questions | `planner.py` |
| Structured output | The model must reply with JSON that matches a Pydantic schema | `gemini.py` (`ask_json`), `extractor.py` |
| Embeddings + cosine similarity | Sentences become vectors; similar meaning gives similarity close to 1.0 | `verifier.py` |
| Source credibility | Each website gets a trust tier from an editable list (a proxy, not proof) | `credibility.py`, `trusted_sources.yaml` |
| Self-critique | A second pass checks that every sentence is backed by a source | `writer.py` |
| Memory / state | All questions, sources, claims and labels are kept in one object and saved as JSON | `models.py`, `pipeline.py` |
| Guardrails | Step and source limits, prompt-injection defence, polite crawling, retry on rate limits | `gemini.py`, `fetch.py`, `pipeline.py` |

## Cross-validation rules

1. **Extract claims.** Each claim records the claim, a quote, the question it answers, its source, and its kind (fact, statistic or opinion). Opinions are never counted as verified facts.
2. **Group matching claims.** Embed claims with `gemini-embedding-2` and group those with cosine similarity ≥ 0.85 under the same research question.
3. **Check agreement.** Gemini labels each source as *supports*, *contradicts* or *partly supports* the group's main claim. It checks numbers, dates and names carefully.
4. **Count independent sources.** Count distinct websites only, and drop near-identical copies, so one story republished five times counts once.
5. **Assign a label.**

   | Label | Rule |
   |---|---|
   | Verified | 2+ independent websites agree, at least one tier 1 or 2, and no tier 1 or 2 source disagrees |
   | Likely | One tier 1 source, or 2+ tier 3 sources agree |
   | Contested | Credible sources disagree, so both sides go in the report |
   | Unverified | Only one low-trust source; run a targeted search (`site:gov`, OpenAlex). If still weak, leave it out of the findings |

The label is an **estimate** of how well a claim is supported, not proof that it is true.

## Trust tiers (`agent/trusted_sources.yaml`, already built)

| Tier | Weight | Examples |
|---|---|---|
| 1 | 0.9 | `.gov`, `.edu`, `.ac.in`, WHO, UN, World Bank, OECD, journal publishers |
| 2 | 0.7 | Reuters, AP, BBC, Britannica; Wikipedia 0.6 (used to find primary sources); arXiv 0.75 (preprints, not peer reviewed) |
| 3 | 0.4 | Any website not listed |
| 4 | 0.1 | Reddit, Quora, social media, hosted blogs; never above 0.2 |

**Adjustments:**
- A named author adds 0.05, and so does a publication date.
- OpenAlex results use their work type: a journal article counts as tier 1, a preprint as tier 2.

## Technology (exact names)

- **Gemini API models** (provider: Google):
  - `gemini-3.5-flash-lite` (main)
  - `gemini-3.1-flash-lite` (bulk extraction, and backup when the main model is busy, slow or out of daily quota)
  - `gemini-embedding-2` (embeddings)

  All are listed as free of charge on the free tier. Flash-Lite was chosen because it has the highest free daily limit. Google's own error reported 20 requests a day for `gemini-3.6-flash`, while third-party guides report about 500 a day for Flash-Lite. Model IDs live in `.env`.
- **SDK:** `google-genai` 2.29.0. Its `models.generate_content` is used with automatic function calling switched off, so our own loop runs every tool.
- **Search:**
  - DuckDuckGo through `ddgs`, an unofficial library.
  - Wikipedia through the MediaWiki Action API and the Wikimedia REST API.
  - Scholarly works through the OpenAlex API, which gives metadata and abstracts but not full text.
- **Reading:** `trafilatura` for HTML, `pypdf` for PDF, `tldextract` for domains.
- **Other:**
  - `pydantic`, `numpy` and `PyYAML` for data.
  - `ReportLab` for the PDF, chosen because it installs with pip alone on Windows.
  - `matplotlib` for charts, `Streamlit` for the UI, `pytest` for tests.
- **Diagrams:** PlantUML 1.2026.1 (UML) and Graphviz 16.1.0 (architecture and DFDs).

## File layout

```
agentic_agent/
├─ agent/
│  ├─ config.py, cache.py, gemini.py
│  ├─ search.py, fetch.py, credibility.py, trusted_sources.yaml
│  ├─ models.py          ResearchState, Question, Source, Claim, ClaimGroup, Report
│  ├─ pipeline.py        runs the 6 phases, streams progress events, resume
│  ├─ planner.py, researcher.py, extractor.py, verifier.py, writer.py
│  └─ pdf_report.py, charts.py
├─ main.py               command line: python main.py "topic" --depth quick
├─ app.py, style.css     Streamlit web page (neo-brutalism design)
├─ run_app.bat           one-click launcher
├─ diagrams/             PlantUML + Graphviz sources and PNG images
├─ media/                web page screenshots and the demo video
├─ *.pdf                 report copies of finished runs
├─ evaluate.py           evaluation runner (planned)
├─ tests/                unit tests, kept locally (not in the repository)
├─ data/cache/           saved responses (not committed)
└─ runs/<id>/            state.json, trace.jsonl, report.pdf (not committed)
```

**Code style rules:**
- Comments are short and single-line only.
- File names are simple and descriptive, with no numbered prefixes.
- Use proper technical terms, explained simply.

## Depth presets (`config.py`)

| Depth | Questions | Max sources | Max agent steps | Expected Gemini calls |
|---|---|---|---|---|
| quick | 4 | 10 | 12 | about 20 |
| standard | 6 | 20 | 25 | about 50 |
| deep | 8 | 35 | 40 | about 100 |

## PDF report layout

1. **Title page:** topic, date, depth, and a note that the report was generated by AI.
2. **Table of contents.**
3. **Executive summary:** key findings, each with a confidence label.
4. **Method:** the questions, the searches run, the number of sources found, read and used, and the trust and validation rules.
5. **Findings:** one section per research question, with numbered citations.
6. **Contested points:** what each side says, and which sources say it.
7. **Source quality:** a tier pie chart, a confidence bar chart, and a table of top sources.
8. **Limitations.**
9. **References:** numbered, with tier, date and access date, and clickable links.
10. **Appendix:** a claim × source table (supports / contradicts).

The writer returns structured JSON (headings, paragraphs, citation IDs), and `pdf_report.py` lays it out directly with ReportLab.

## Safety and guardrails

- **Prompt injection:**
  - Tool results are wrapped as `untrusted_data`.
  - Every prompt says never to follow instructions found inside sources.
  - Extraction only accepts JSON that fits the schema.
- **Polite crawling:**
  - A clear User-Agent and a 20-second timeout.
  - An 8 MB size cap.
  - At least 1 second between requests to the same website.
  - Paywalls and login pages are skipped.
- **Copyright:** the report summarises in its own words and only uses short quotes with citations.
- **Hallucination control:** findings come only from extracted claims, and the critic removes uncited sentences.
- **Budget:**
  - Step, source and call limits.
  - Exponential backoff on HTTP 429.
  - A disk cache, so repeated runs cost less.

## Evaluation (`evaluate.py`)

- **Topics:** 8–10, kept small for the free tier:
  - Factual topics with known key facts.
  - Myth topics with a known answer, e.g. "Is the Great Wall of China visible from space?".
  - One fast-moving news topic.
- **Systems compared:**
  1. Plain Gemini, one call, no tools (baseline).
  2. The agent without cross-validation.
  3. The full agent.
- **Metrics:**
  - Key-fact coverage.
  - Citation precision, judged by Gemini with the source text in front of it.
  - Myth detection rate.
  - Percentage of tier 1 and 2 sources, and number of unique websites.
  - A manual check of a sample of Verified claims.
  - Gemini calls and time per report.
- **Output:** a CSV and a markdown table, plus 2–3 example PDFs for the report appendix.

## Milestones

| # | Work | Status |
|---|---|---|
| 1 | Setup, `gemini.py` (retry, model fallback, daily-quota handling, JSON output, embeddings, tool messages), `cache.py` | Done |
| 2 | `search.py`, `fetch.py`, `credibility.py` | Done |
| 3 | `models.py`, `planner.py`, `researcher.py`, `pipeline.py` (with resume), `main.py` | Done |
| 4 | `extractor.py` (grounding check), `verifier.py` | Done |
| 5 | `writer.py` (rule-based critic), `pdf_report.py`, `charts.py` | Done |
| 6 | `app.py` (Streamlit web page, neo-brutalism design), `run_app.bat`, PDF copy in the main folder | Done, 73 unit tests passing, checked live |
| 7 | `evaluate.py` and results (include the myth topics) | Next |
| 8 | Project documentation (README), 11 diagrams, screenshots, demo video, 3 test cases | Done |

## Verification

- `python -m pytest` runs all unit tests with fake responses, so it needs no internet and no API key.
- `python main.py "Microplastics in drinking water: health effects" --depth quick` should create `runs/<id>/report.pdf`. Check that:
  - Citations match the reference list.
  - Contested claims have their own section.
- On a myth topic, the false claim must come out Contested or be rejected, never Verified.
- `streamlit run app.py` shows the phases live and lets you download the PDF.
