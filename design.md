# AnyRAG — Design

A generic, fully local RAG desktop app: point it at any folder of documents, get grounded answers with citations. Direct generalization of **AskJ** (see `../AskJ/design.md`) — same staged pipeline and retrieval core, with everything Sison-specific replaced by per-collection configuration.

## Goals

- **Any corpus:** user creates named **collections** (e.g. "Tax docs", "Research papers"), each with its own sources, index, and settings. Switch collections in the app.
- Grounded answers with citations (document title, section, page); clickable citations open a source reader.
- Answer modes: **Scholar** (neutral, always available), **Quotes** (verbatim, mechanically verified), **Persona** (opt-in per collection when the corpus is a single author's voice).
- 100% local: no API keys, nothing leaves the machine. Configurable LLM backend incl. tiny models.
- Ships as a desktop app; also usable as CLI and local web UI.

## What generalizes from AskJ (and how)

| AskJ (single hardcoded corpus) | AnyRAG (generic) |
|---|---|
| `SOURCE_META` dict in `config.py` | Per-collection `collection.json` + per-source metadata, auto-detected then user-editable in the UI |
| `AUTHOR` / `COLLECTION` constants in prompts | Prompt templates rendered from collection settings (`name`, `author`, `description`) |
| Persona mode always offered | Persona is per-collection opt-in (`persona: {enabled, author, disclaimer}`); hidden otherwise |
| `writings/` + one `data/` tree | `collections/<id>/sources/` + `collections/<id>/data/` per collection |
| Timeline assumes dated essays | Date extraction runs opportunistically; timeline UI appears only when coverage > threshold (~40% of sections dated) |
| PDF/txt/md | + docx (via `python-docx`), html (strip tags), epub (later) |
| TOC parsing tuned per volume | Generic segmentation ladder: PDF outline/bookmarks → numbered-TOC parse → heading heuristics (ALL-CAPS, markdown `#`, docx styles) → whole-file fallback |

Everything else — staged pipeline, content-hash staleness, hybrid retrieval, hierarchical summaries, quote verification, source reader, conversations — ports unchanged in logic, re-scoped per collection.

## Architecture

Per collection, the same staged pipeline as AskJ; query time degrades gracefully (no vectors → BM25; no summaries → flat retrieval).

```
collections/<id>/sources/*.{pdf,txt,md,docx,html}
     │  STAGE 1  CHUNK      extract → segment (outline/TOC/heading ladder)
     │                      → clean → chunk + section/page metadata
     │           sidecars:  paras (1a) · dates (1c) · quote bank (1d)
     ▼
data/chunks/<source>.jsonl
     │  STAGE 1b SUMMARIZE  local LLM abstract + keywords per section
     ▼
data/summaries/<source>.jsonl (+ .src hash sidecar)
     │  STAGE 2  EMBED      chunks AND summaries; vectors tagged
     │                      {model, sha256(source artifact)}
     ▼
data/vectors/<source>.npz (+ .summaries.npz)
     │  RETRIEVE  explore (summaries) → exploit (BM25+dense RRF, weighted)
     │            → optional rerank → per-source diversity cap → relevance gate
     ▼
LLM backend (ollama | openai-compatible | embedded llama.cpp), streaming
     ▼
answer + citations → CLI / web UI / desktop shell (source reader, quote ✓/⚠)
```

Invariants carried over verbatim from AskJ: chunks before summaries, both as text before any embedding; staleness is content-based (sha256 + model tag), never mtime; sidecars never touch chunk bytes; per-collection `data/manifest.json`; `--force` re-processes; parallel ingestion (process pool for chunking, concurrent LLM calls with per-section resume for summaries, shared batched encoder for embedding).

## Collections

**Registry:** `USER_ROOT/collections.json` — `[{id, name, path, created}]`. Each collection directory:

```
collections/<id>/
├── collection.json      # name, description, author?, persona?, source_meta{}
├── settings.json        # per-collection overrides (chunking, retrieval, mode defaults)
├── sources/             # the documents (or symlinks/refs to a user folder)
└── data/                # chunks/ summaries/ vectors/ paras/ dates/ quotes/
                         # conversations/ manifest.json
```

Two source modes: **managed** (files copied/imported into `sources/`) and **linked** (collection points at an existing folder; ingest watches for adds/changes on demand via a Rescan button — content hashing makes rescans cheap). Linked is the default for "point at my folder" UX; managed for drag-and-drop imports.

**Per-source metadata** (`source_meta` in `collection.json`, same shape as AskJ's `SOURCE_META`): `title`, `years`, `page_offset`, `toc_pdf_pages`. Auto-populated at first ingest (PDF metadata/outline, filename heuristics), editable in a small UI table. Absent metadata = generic chunking, filename as title — exactly AskJ's fallback.

**Isolation:** indexes, conversations, and settings never cross collections. One in-memory index is loaded at a time (LRU cache of 2–3 for fast switching); the active collection id travels in every API call.

## Retrieval & generation

Identical to AskJ's `rag.py` core, parameterized per collection: BM25 (stdlib, always available) / dense / hybrid RRF; hierarchical summaries-first when summaries exist (`HIERARCHICAL=auto`); optional cross-encoder rerank; per-source diversity cap; score/coverage gate → "not covered in this collection" without an LLM call. Embedder default `BAAI/bge-small-en-v1.5` (AskJ's planned upgrade — start there, 384-dim, ONNX available); model tag in vectors forces re-embed on switch.

**Prompts:** three system templates with `{collection_name}`, `{description}`, `{author}` slots. Scholar is corpus-neutral ("answer only from the excerpts; cite [Title — Source, p. N]"). Quotes mode + post-generation verbatim verification (normalize → substring → fuzzy ≥ 0.9 → `verify` NDJSON event, ✓/⚠ in UI) is corpus-independent and ports as-is. Persona renders only when enabled, always with the AI-reconstruction disclaimer.

**Backends:** `ollama` | `openai` (any OpenAI-compatible local server) | `embedded` (llama-cpp-python + bundled GGUF, desktop default). Streaming everywhere; HTTP via stdlib `urllib`.

## Server & API

Threaded stdlib HTTP server (as AskJ's `serve.py`), all routes collection-scoped:

```
GET  /collections                      list · POST create · DELETE remove
POST /c/<id>/ingest                    (re)ingest; NDJSON progress events
GET  /c/<id>/stats                     sources, chunks, vectors, summary coverage
POST /c/<id>/ask                       NDJSON stream: meta, tokens, sources, verify
GET  /c/<id>/passage/<chunk_id>        windowed/full stitched passage (reader)
GET  /c/<id>/read?id=…                 full-section reader page
GET  /c/<id>/conversations …           history (AskJ Option A + export)
GET  /c/<id>/timeline · /quote         only when data supports them
GET  /settings · /c/<id>/settings      global + per-collection
```

Ingest progress streams as NDJSON (`{stage, source, done, total}`) so the UI can show a real progress bar — the one genuinely new server feature vs AskJ (whose ingest was CLI-only).

## Desktop app — two shells, one backend

The product is the Python backend + single-file web UI. Shells are thin:

1. **pywebview (v1, default):** AskJ's `app.py` model — server on a random localhost port, native window, PyInstaller one-folder bundle with embedded GGUF + ONNX query encoder (~1.3 GB, budget as AskJ's `distribution.md`). Ships first.
2. **Electron (v2, optional):** same localhost server as a spawned sidecar; Electron adds native menus, auto-update, tray, and nicer installers. Adopted only if pywebview's window proves limiting — the UI is identical HTML either way, so nothing is rewritten.

First-run UX: welcome screen → "Create your first collection" → pick a folder → ingest with progress → starter questions. Model lazy-loads on first question ("warming up" notice).

## UI additions vs AskJ

- **Collection switcher** (sidebar or header dropdown) with per-collection stats.
- **New-collection wizard:** name → folder picker (linked) or file drop (managed) → optional author/persona → ingest progress.
- **Source manager table:** per-source title/years/status, edit metadata, re-ingest, remove.
- Everything else (chat + mode toggle, citations → reader modal, history panel, quote cards, timeline) is AskJ's UI re-skinned, shown/hidden by collection capability flags from `/stats`.

## Project layout

```
AnyRAG/
├── design.md  plan.md
├── setup.sh  requirements.txt  requirements-app.txt
├── config.py  paths.py  collections.py        # registry + Collection class
├── ingest.py  summarize.py  dates.py  quotes_bank.py
├── rag.py  conversations.py
├── ask.py  serve.py  app.py  eval.py
├── static/index.html
├── anyrag.spec  scripts/  .github/workflows/
└── (user data lives in platformdirs user-data dir, incl. collections/)
```

Note: unlike AskJ, **all** mutable data lives in the user-data dir from day one (`platformdirs`) — no bundle/user split headaches later, and dev mode just points `USER_ROOT` at `./devdata/`.

## Key decisions & trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Port AskJ rather than rewrite | ~2,850 lines of proven pipeline/retrieval/verification code | Carries some single-corpus assumptions to unwind (audit in plan Phase 1) |
| Collections as directories + JSON registry | Zero-dependency, inspectable, trivially portable/deletable | No cross-collection search (out of scope v1) |
| Linked-folder sources with hash-based rescan | "Point at my folder" is the core generic UX | No live file watching v1; user clicks Rescan |
| Generic segmentation ladder, metadata editable | Can't hand-tune TOCs per corpus like AskJ did | Worse section titles on messy PDFs; UI edit is the escape hatch |
| Capability-gated features (timeline, persona) | Generic corpora often lack dates/single author | Feature discovery is subtler |
| pywebview first, Electron later, same server | One backend, shell is swappable | v1 window lacks native menus/auto-update |
| bge-small-en-v1.5 from the start | AskJ's measured upgrade path; ONNX exists | ~100 MB encoder in bundle |

## Non-goals (v1)

Cross-collection queries; live folder watching; OCR of scanned PDFs (flag them, skip); cloud backends; multi-user; mobile.
