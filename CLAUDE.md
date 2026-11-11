# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AnyRAG: a fully local, generic RAG app — point it at any folder of documents
(PDF/txt/md/docx/html) and get grounded answers with citations. No API keys,
nothing leaves the machine. It's a generalization of a sibling project
`../AskJ/` (single hardcoded corpus) into a multi-**collection** app where
every corpus-specific thing (name, author, persona, per-source metadata) is
config on a `Collection` object instead of a module-level global.

Read `design.md` (architecture/rationale) and `plan.md` (phased build status,
what's verified vs. not yet tested) before making non-trivial changes —
they're kept current and are more detailed than this file.

## Commands

All Python runs through the repo-local venv — never system Python, never
another project's venv:

```bash
bash setup.sh                                 # one-time: .venv + requirements.txt

# collections
.venv/bin/python -c "import library; library.create('My docs', link='/path/to/docs')"
.venv/bin/python ask.py --list                # list collections

# pipeline (chunk [+ summarize/embed if deps installed])
.venv/bin/python ingest.py -c <collection-id> [--force]

# query
.venv/bin/python ask.py -c <collection-id> "question" [--mode scholar|persona|quotes] [--chat]
.venv/bin/python ask.py -c <collection-id> --quote [--topic X]

# run
.venv/bin/python serve.py                     # web UI at http://127.0.0.1:8000
.venv/bin/python app.py                        # desktop window (pywebview), falls back to browser

# retrieval quality (per-collection data/eval.jsonl of {"q", "expect"} pairs)
.venv/bin/python eval.py -c <collection-id> [--compare] [-k 10]

# desktop packaging (fetches GGUF + exports ONNX encoder, then PyInstaller)
bash scripts/build_desktop.sh
```

Tests: `pytest` (dev-only, `requirements-dev.txt`) runs against a temp
`ANYRAG_HOME` and a generated fixture corpus (`tests/fixtures/build.py`);
known defects are strict-xfail tests in `tests/test_known_defects.py`.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # once
.venv/bin/python -m pytest -q
bash scripts/verify.sh M<n>        # milestone gate (implem-plan-enhancements.md)
```

There is no linter or formatter configured — don't invent commands for
them. `ANYRAG_HOME` overrides the user-data dir
(defaults to `./devdata/` in dev); useful for testing against a clean or
alternate data directory without touching real collections.

Dense embedding / hybrid retrieval and the reranker are optional heavy deps
(`sentence-transformers`, `numpy`) commented out in `requirements.txt` —
BM25-only works with just `pypdf` + `python-docx`. Uncomment, re-run
`setup.sh`, then `ingest.py --force` to enable.

## Architecture

### Everything is scoped to a Collection

`library.py` holds the registry (`USER_ROOT/collections.json`) and the
`Collection` class, which is the object every other module operates on
(`col.chunks_dir`, `col.vectors_dir`, `col.source_meta`, `col.setting(key)`,
etc.). There are no more corpus-specific globals like AskJ's `SOURCE_META`/
`AUTHOR` — anything corpus-specific lives in `collection.json` /
`settings.json` under the collection's own directory tree:

```
collections/<id>/
├── collection.json      # name, description, author?, persona?, source_meta{}
├── settings.json        # per-collection overrides of config.py defaults
├── sources/              # managed docs (or nothing, if "linked")
└── data/                 # chunks/ paras/ dates/ quotes/ summaries/ vectors/
                          # conversations/ manifest.json eval.jsonl
```

Two source modes: **managed** (files copied into `sources/`) and **linked**
(`collection.json["link"]` points at an existing folder the user owns —
never written to; ingest rescans it via content hashing). `config.py` now
holds only global defaults + LLM backend settings; `paths.py` resolves the
bundle dir (read-only, PyInstaller-aware) vs. the user-data dir (all mutable
state — `platformdirs`-style, `ANYRAG_HOME` overrides it in dev).

### Staged pipeline (per collection, driven by `ingest.py`)

```
sources/*.{pdf,txt,md,docx,html}
  → STAGE 1  chunk   (ingest.py): segmentation ladder — PDF outline/bookmarks
                      → numbered-TOC parse → heading heuristics → whole-file
                      fallback — then clean + chunk with section/page metadata.
                      Sidecars alongside: paras (display), dates (dates.py),
                      quote bank (quotes_bank.py).
  → data/chunks/<source>.jsonl
  → STAGE 1b summarize (summarize.py, optional) — local-LLM per-section abstract
  → data/summaries/<source>.jsonl (+ .src hash sidecar)
  → STAGE 2  embed    (ingest.py) — chunks AND summaries embedded; vectors
                      tagged with {model, sha256(source)} for staleness checks
  → data/vectors/<source>.npz (+ .summaries.npz)
```

Staleness is always content-hash based (sha256 + embedding-model tag), never
mtime. `--force` reprocesses. Ingestion parallelizes: a process pool for
chunking, concurrent LLM calls with per-section resume for summaries, a
shared batched encoder for embedding. `manifest.json` tracks per-source
pipeline state.

### Retrieval & generation (`rag.py`, the largest module)

Hybrid retrieval: pure-Python `BM25Index` (stdlib-only, always available)
fused with `DenseIndex` (sentence-transformers or the ONNX query encoder)
via reciprocal-rank-fusion in `HybridIndex`; automatic fallback to BM25 alone
when embedding deps/vectors are absent (`RETRIEVAL = "auto"` in `config.py`).
When summaries exist, `HierarchicalIndex` runs an explore-then-exploit pass
(select top sections by summary match, boost chunks within them) before
falling back to flat retrieval. An optional cross-encoder reranks the final
candidate pool (`RERANK = "auto"`). `diversify()` caps chunks per source so
one large document can't crowd out the rest of a collection. `gate_ok()` is
a pre-LLM relevance gate — score/coverage too low → "not covered in this
collection" answered without a model call; the gate scales with corpus size
(`log(n_docs)`) rather than a fixed threshold, since a fixed `MIN_SCORE` was
found to unfairly block small collections.

Three answer modes, rendered from collection settings (never hardcoded
author/corpus name): **scholar** (neutral, cites `[Section, p. N]`),
**quotes** (verbatim, mechanically verified — normalize → substring → fuzzy
match ≥ `QUOTE_MATCH_RATIO` → ✓/⚠ per quote via `verify_quotes()`), and
**persona** (first-person, opt-in per collection via `collection.json`,
requires an `author` to be set, always rendered with an AI-reconstruction
disclaimer). `Runtime` (bottom of `rag.py`) is the stateful per-collection
object `ask.py`/`serve.py` hold: loaded index + `ask_stream()`/`ask()`.

**LLM backends** (`config.py` → `LLM_BACKEND`, overridable per-user via
`settings.json`): `ollama` (default, native API), `openai` (any
OpenAI-compatible local server — llama.cpp server, LM Studio, vLLM), or
`embedded` (llama-cpp-python loading a bundled GGUF in-process — the desktop
build's default, no server needed). All streaming, all via stdlib `urllib`
— no HTTP client dependency.

### Server & desktop shell

`serve.py` is a threaded stdlib `BaseHTTPRequestHandler`; every route is
collection-scoped under `/c/<id>/…` (see the `_route()`/dispatch methods)
plus top-level `/collections` and `/settings`. Ingest streams progress as
NDJSON (`{stage, source, done, total}`) so the UI can show a real bar; `/ask`
streams NDJSON too (`meta`, token chunks, `sources`, `verify`). `static/
index.html` is the entire web UI (single file, vanilla — no build step).

`app.py` is the desktop shell: starts `serve.py` on a random localhost port
in a background thread, opens a native `pywebview` window pointed at it
(falls back to opening the default browser if `pywebview` isn't installed).
Same backend either way — the web UI and desktop app are the same server.
`anyrag.spec` (PyInstaller, one-folder build) bundles `static/` and
`models/` (GGUF + ONNX encoder, fetched by `scripts/fetch_model.py` /
`scripts/export_onnx.py`) but deliberately **no corpus** — the app ships
empty (~1.2 GB with the embedded Qwen2.5-1.5B model); all data a user
creates lives in the platform user-data dir, never in the bundle.

### Conventions carried through every stage

- Chunks are produced and persisted as text *before* summaries, and both
  exist as text before anything is embedded — never skip straight to
  vectors.
- Sidecars (`paras`, `dates`, `quotes`, `summaries`) never mutate chunk
  bytes; they're keyed alongside them.
- Section headings are stripped from chunk bodies at segmentation time but
  indexed together with the chunk text in `rag.py`'s `tokenize()` — a
  section titled "Deep Sea Mining" must still match that query even though
  the heading isn't repeated in the body.
- When changing retrieval scoring/gating, re-run `eval.py --compare` on a
  real collection and record the before/after numbers in `plan.md` (its own
  documented discipline: "measure, don't guess") rather than reasoning about
  it in the abstract.
