# AnyRAG

Fully local question-answering over **any** documents. Create collections
from folders of PDFs, Word files, Markdown, HTML, or text; ask in natural
language; get grounded answers with citations that open a source reader.
No API keys; nothing leaves your machine. Generalized from
[AskJ](../AskJ/) — same staged pipeline, per-collection everything.

## How it works

Each collection runs the staged pipeline: sources are **chunked** (with a
segmentation ladder: PDF outline → TOC parse → title match → headings →
whole-file), sidecars extract **dates** and a **quote bank**, an optional
local-LLM pass writes per-section **summaries**, then chunks and summaries
are **embedded**. Retrieval is hybrid — pure-Python BM25 fused with dense
vectors via RRF, hierarchical when summaries exist — with automatic
fallback to BM25 alone. Generation streams from Ollama, any
OpenAI-compatible server, or an embedded llama.cpp model (desktop build).
Details in `design.md`; build plan in `plan.md`.

## Quickstart

```bash
bash setup.sh                                # one-time: .venv + deps

# create a collection from a folder of documents:
.venv/bin/python -c "import library; library.create('My docs', link='/path/to/docs')"
.venv/bin/python ingest.py -c my-docs        # chunk (+summarize/embed if available)

# pick a backend in config.py (default: Ollama + llama3.1:8b), then:
.venv/bin/python ask.py -c my-docs "What do these documents say about X?"
.venv/bin/python serve.py                    # web UI → http://127.0.0.1:8000
.venv/bin/python app.py                      # desktop window (pywebview)
```

The web UI can do all of it without the shell: create/link collections,
watch ingest progress, import files, edit source titles, switch collections.

## Answer modes

**Scholar** (neutral, cites `[Section, p. N]`), **Quotes** (verbatim
quotations, mechanically verified ✓/⚠ against the sources), and
**Persona** (first-person voice — opt-in per collection, only when the
corpus has a named author; clearly tagged as an AI reconstruction).

## Desktop app

```bash
bash scripts/build_desktop.sh     # GGUF + ONNX encoder → dist/AnyRAG.app
```

Ships empty (~1.2 GB with the embedded Qwen2.5-1.5B model) — users create
collections from their own folders; all data lives in the platform
user-data dir.

## Repo notes

All mutable data (collections, indexes, conversations, settings) lives in
the user-data dir — `./devdata/` in dev, `ANYRAG_HOME` to override.
`models/` is fetched by scripts and git-ignored. All Python runs through
the repo-local `.venv`.

## Files

`library.py` collection registry + Collection class · `config.py` global
tuning + backend (+ per-user `settings.json`) · `ingest.py` chunk→embed
pipeline + segmentation ladder · `summarize.py` `dates.py` `quotes_bank.py`
pipeline stages · `rag.py` retrieval + generation + Runtime ·
`conversations.py` saved history · `ask.py` CLI · `serve.py` web server ·
`app.py` desktop shell · `eval.py` per-collection retrieval eval ·
`anyrag.spec` + `scripts/` desktop packaging
