# AnyRAG — Build Plan

Phased port-and-generalize of AskJ (`../AskJ/`). Each phase ends runnable with a verification step; ship order optimizes for a usable generic pipeline early, desktop polish late.

## Status — 2026-07-16

Phases 0–4 built and smoke-tested; Phase 5 packaging files ready (build itself needs the target OS). Verified end-to-end in a sandbox: two linked collections (md / txt / html) ingest and stay fully isolated (retrieval, gate, conversations); a third mixed collection ingested a 400-page PDF that auto-segmented into 33 titled articles via the `toc` ladder rung — the same book AskJ needed hand-tuned `toc_pdf_pages` for — plus a .docx segmented on Heading styles; citations carry printed page numbers; quote verification flags a fabricated quote ⚠ and passes a verbatim one ✓; the full HTTP API (create/import/ingest-stream/ask-stream/passage/quote/source-meta/conversations/delete) answers correctly, with gated questions answered without an LLM and backend-unreachable surfaced as a stream error event. Not yet tested: generation with a live LLM, dense/hybrid retrieval (needs sentence-transformers), summaries stage, desktop build.

Two generic-corpus fixes discovered by testing (now in `rag.py`): the relevance gate's absolute `MIN_SCORE` blocked small collections — it now scales with log(corpus size); and section titles are indexed with chunk text (headings are stripped from bodies at segmentation, so "deep sea mining" must match a section titled that). Plus light plural folding in `tokenize`, and TOC-title cleanup in `ingest.py` (extraction-artifact `�` chars stripped, over-merged titles >120 chars rejected).

The whole suite was re-verified through a dedicated virtualenv (`setup.sh`; `VENV_DIR` overrides the location) with only `requirements.txt` core deps installed — pypdf + python-docx suffice for BM25-only operation; the embedding stack stays commented out as optional. Known gate limitation carried from AskJ: a 2-term question where one common term matches ("capital of Mongolia" vs a book discussing capital) can squeak past the BM25 coverage check at exactly 0.5 — the dense gate tightens this once vectors exist.

## Phase 0 — Scaffold & port audit (½ day)

- Repo scaffold per `design.md` layout; `setup.sh` + repo-local `.venv`.
- Copy AskJ modules verbatim: `ingest.py`, `summarize.py`, `rag.py`, `dates.py`, `quotes_bank.py`, `conversations.py`, `ask.py`, `serve.py`, `app.py`, `eval.py`, `static/index.html`, `paths.py`.
- **Audit pass:** grep for every Sison-specific symbol (`AUTHOR`, `COLLECTION`, `SOURCE_META`, `WRITINGS_DIR`, persona prompt text, "Sison", "writings") → checklist of call sites to generalize. This checklist drives Phases 1–3.

✅ *Verify:* copied code runs AskJ's corpus unmodified from the new repo (smoke: `ask.py` one question).

## Phase 1 — Collection abstraction (1–2 days)

- `collections.py`: registry (`USER_ROOT/collections.json`) + `Collection` class exposing what `config.py` used to export globally: paths (`chunks_dir`, `vectors_dir`, …), `source_meta`, merged settings (global defaults ← collection `settings.json`).
- `config.py` shrinks to global defaults + backend settings; all path/corpus globals removed. `paths.py` → platformdirs user-data dir always; `ANYRAG_HOME` env override for dev.
- Thread a `collection` parameter through `ingest.py` and `rag.py` (mechanical: replace module-global path/meta lookups). Manifest, hashes, sidecars unchanged.
- Managed vs linked source modes; `rescan()` = re-run ingest (content hashing already makes it incremental).

✅ *Verify:* two collections side by side (AskJ corpus as one, a folder of unrelated PDFs as the other); ingest both; `ask.py -c <id>` answers from the right corpus with zero cross-contamination; pytest for registry CRUD + path isolation.

## Phase 2 — Generic ingestion (2–3 days)

- **Segmentation ladder** in `ingest.py`: PDF outline/bookmarks → numbered-TOC parse (existing AskJ code, made metadata-driven) → heading heuristics (ALL-CAPS lines, markdown `#`, docx styles) → whole-file fallback. Each source records which rung it used (surfaced in source manager).
- New extractors: docx (`python-docx`), html (stdlib `html.parser` strip). Keep pdf/txt/md.
- **Metadata auto-detection:** title from PDF metadata/first heading/filename; `years` from filename/content regex (reuse `dates.py` guards). Written into `collection.json.source_meta`, user-editable later.
- Ingest as a callable with a progress callback (stage, source, done/total) — needed by Phase 4's NDJSON endpoint; CLI prints the same events.

✅ *Verify:* ingest a deliberately messy mixed folder (scanned-ish PDF, docx, md, html); every source chunks without error; segmentation rung recorded; spot-check section titles on 3 sources.

## Phase 3 — Prompts, modes, capability gating (1 day)

- Three prompt templates with `{collection_name}/{description}/{author}` slots; scholar neutral by default.
- Persona: renders only when `collection.json.persona.enabled`; quotes mode + verification ports as-is (corpus-independent).
- Capability flags computed from data: `has_vectors`, `has_summaries`, `date_coverage`, `persona` → returned by `/stats`, consumed by UI and CLI (e.g. timeline hidden below 40% date coverage).

✅ *Verify:* quotes mode on a non-Sison collection produces verified ✓ quotes; persona absent on collections without it; scholar citations use auto-detected titles.

## Phase 4 — Server & UI (2–3 days)

- Namespace all routes under `/c/<id>/…`; add `/collections` CRUD, `POST /c/<id>/ingest` (NDJSON progress), per-collection `/settings`.
- UI: collection switcher; new-collection wizard (name → folder pick/link or file drop → optional persona → live ingest progress); source-manager table (title/years edit, re-ingest, remove, segmentation rung shown); capability-gated panels (timeline, quote card, persona toggle).
- Conversations (AskJ Option A + Markdown/JSON export) scoped per collection — mostly path re-plumbing.

✅ *Verify:* full flow in the browser: create collection from a folder → watch progress → ask → click citation → reader opens → history persists per collection; switch collections and confirm isolation.

## Phase 5 — Desktop v1: pywebview bundle (2–3 days)

- `app.py`: server on random port + pywebview window (AskJ pattern); native folder-picker dialog wired to the wizard.
- PyInstaller one-folder build (`anyrag.spec`): embedded Qwen2.5-1.5B GGUF, ONNX query encoder (bge-small), no torch at runtime (sentence-transformers dev-only, as AskJ `distribution.md`). Unlike AskJ, **no corpus bundled** — app ships empty, ~1.2 GB.
- CI matrix (macOS/Win/Linux) → dmg / Inno Setup / AppImage; smoke test: launch, create collection from fixture docs, POST /ask.

✅ *Verify:* clean machine (or VM): install, first-run wizard on a fresh folder, answer with citations, quit/relaunch → collection persists in user-data dir.

## Phase 6 — Eval & polish (ongoing)

- `eval.py` generalized: per-collection question file (`data/eval.jsonl`), top-k section-hit rate; run flat vs hierarchical vs +rerank, keep what moves the number (AskJ's S5 discipline).
- Reranker (`ms-marco-MiniLM-L-6-v2`, ONNX) behind `RERANK=auto`.
- Settings UI (backend, model, temperature, chunk size per collection).

✅ *Verify:* eval numbers recorded in this file per milestone; no regression vs AskJ's 18/20 on its own corpus.

## Phase 7 — Electron shell (optional, later)

Only if pywebview limits bite (menus, auto-update, installer polish). Electron spawns the same Python server as a sidecar; UI unchanged. Decision point: after v1 user feedback.

## Risks

| Risk | Mitigation |
|---|---|
| Generic segmentation poor on messy PDFs | Ladder + rung visibility + editable metadata; whole-file fallback always works |
| Summary stage slow on big generic corpora | Stage 1b optional (flat retrieval fallback); resumable per-section checkpointing already ported |
| Port carries hidden single-corpus assumptions | Phase 0 audit checklist; Phase 1 two-collection isolation test |
| Bundle size / model licensing | Qwen Apache-2.0; no corpus shipped, so no rights questions at all |

**Total estimate: ~2 weeks to desktop v1 (end of Phase 5).**
