# Changelog

## 2026-11-12

### Bug fix — M5 goal text in the implementation plan could not complete
The M5 section of `implem-plan-enhancements.md` gave a goal template that the
`/goal` checker could never mark done: it was pasted with `M5-<n>` unfilled,
it made "with one subagent" part of the completion condition (not checkable,
and unsatisfiable once an agent is stopped), and it treated a revert as
"stop" instead of a valid outcome. Rewrote the section: concrete goal texts
per item whose condition is only the `verify.sh` line and a `plan.md`
section with a verdict; the subagent and snapshot-revert method moved to
notes; removed the claim that `verify.sh` reads `EVAL_COLLECTION`; M5-6's
two prerequisites stated; added an as-run results table for M5-1…M5-6.

### Feature — M5-2…M5-6: search-only mode, segmentation, chunking, Unicode tokenizer
Remaining M5 goals of `implem-plan-enhancements.md`, run by one subagent,
each item kept only under its "keep if" rule. `bash scripts/verify.sh M5-<n>`
passes for every item; numbers are in `plan.md` ("M5-2 eval" … "M5-6 eval"),
all measured on the fixtures and IBON only (no real multi-document
collection yet).
- **M5-2 (P2-9)** (`rag.py`, `serve.py`, `static/index.html`): kept
  "closest material" (a gated-out question shows the 3 nearest passages)
  and search-only mode (`POST /c/<id>/search` plus a Search button, no model
  call). **Reverted** the idf-weighted gate coverage: false accepts went
  down (1 → 0 hybrid) but false rejects went up (0 → 2 hybrid, 2 → 9
  BM25-only) on 180 labelled pairs. Gate decisions are unchanged.
- **M5-3 (P2-3)** (`ingest.py`): heading detection for PDF (font size and
  bold), HTML, TXT and DOCX; sources left on segmentation rung `none` fell
  10 → 8; hit@k unchanged; IBON 14 → 15 chunks. Sources re-chunk once
  (`SEGMENTER_VERSION`).
- **M5-4 (P2-4)** (`ingest.py`, `rag.py`): chunk ends move back to a
  paragraph or sentence break, at most 10% of `CHUNK_WORDS`; chunks store
  `start`/`end` offsets; `passage()` round trip still exact; hit@k unchanged.
- **M5-5 (P2-6)** (`rag.py`): NFKC, casefolding and Unicode word
  characters in the tokenizer, plus an optional per-collection
  `stopwords.txt`; English results identical; token caches rebuild once.
  Known limit pinned as a strict xfail: the English encoder lets off-topic
  Greek questions through the hybrid gate.
- **M5-6 (P2-7)**: not run. The ONNX reranker needs a model download, which
  needs your approval, and its keep-if rule needs a real collection.
- **M5-1 re-check** by the same subagent on the final code (`rag.py`):
  per-collection stopwords now also apply to the BM25 index built for
  dense-only gating; an ASCII fast path in the tokenizer brings the uncached
  50k-chunk build back to 3.15 s (it was 4.89 s after M5-5), with the same
  tokens. Rankings still identical; search about 110× faster.
- Tests: `tests/test_m5_2_search.py`, `test_m5_3_segmentation.py`,
  `test_m5_4_chunking.py`, `test_m5_5_tokenizer.py`, plus 3 more in
  `test_m5_1_bm25.py` (291 passed, 1 xfailed).

### Feature — M5-1: BM25 that scales (inverted index, gate reuse, token cache)
Goal M5-1 of `implem-plan-enhancements.md` (P2-5). Kept under the plan's
rule: rankings identical on the fixtures, faster on a synthetic 50k-chunk
corpus. Numbers are in `plan.md` ("M5-1 eval"); `bash scripts/verify.sh M5-1`
passes.
- `rag.py`: `BM25Index` scores through an inverted index (term → postings)
  with the old float expression and tie order, so scores and rankings are
  unchanged. Per-query memo; `gate_ok`/`bm25_gate_ok` can reuse the caller's
  hits. Search at 50k chunks: 28 ms → 0.3 ms median.
- Token cache `data/cache/tokens/<artifact key>.json` keyed by chunk-text
  hash plus tokenizer version/stopwords; rebuilt silently when stale or
  corrupt; files for removed sources dropped on load. Warm index load
  slightly faster at 50k chunks (2.6–3.0 s → 2.2–2.5 s); the first load after a change is slower
  (it writes the cache).
- Independent review: `search_idx` with k ≤ 0 now matches the old slice;
  warm-load gain is small and varies between runs (see `plan.md`).
- Tests: `tests/test_m5_1_bm25.py` (16, compares against a verbatim copy
  of the old scorer); 236 passed.

### Feature — M4: settings, source management, background ingest, chat UX
Milestone M4 of `implem-plan-enhancements.md` (P3-1…P3-5), run as a backend
subagent then a UI subagent. `bash scripts/verify.sh M4` passes; fixture
eval identical to M3.
- **Settings** (`config.py`, `serve.py`, `library.py`, `rag.py`):
  `GET|POST /settings` and `/c/<id>/settings` validated against one
  allow-list (`config.SETTINGS_SCHEMA`, types/ranges, null resets),
  applied without restart through `config.get()`; `GET /backend/status`
  (reachability, model presence, Ollama model list, 2 s timeouts).
- **Collections and sources** (`serve.py`, `library.py`, `ingest.py`,
  `summarize.py`): `PATCH /c/<id>`; `DELETE /c/<id>/source/<name>`
  (managed only — linked folders are never written); single-source
  re-ingest; recursive linked-folder scan with ignore patterns, nested
  sources keyed by relative path; per-source `error`/`no_text` in `/stats`;
  sizes in `/collections`.
- **Background ingest** (`serve.py`, `ingest.py`): one job per collection,
  per-stage done/total, `/ingest/status`, `/ingest/cancel` (cooperative,
  interrupted sources redone next run), second clients re-attach with a
  replay of events.
- **UI** (`static/index.html`, `app.py`): settings panel with Test
  connection and Ollama model dropdown; edit-collection form and
  per-collection tuning; remove/re-ingest buttons and error/no-text badges;
  progress bar with Cancel and re-attach; safe Markdown rendering of
  answers (DOM only, http/https links only); Stop button; Copy answer;
  native folder picker in the desktop window.
- Tests: `tests/test_m4a_settings.py`, `test_m4a_sources.py`,
  `test_m4a_jobs.py` (220 passed); baseline
  `tests/fixtures/baselines/M4.json`.

### Feature — M3: follow-up questions, numbered citations, per-collection tuning
Milestone M3 of `implem-plan-enhancements.md` (P2-1, P2-2, P2-8), run as a
backend subagent then a UI subagent. `bash scripts/verify.sh M3` passes;
fixture eval identical to M2 (defaults unchanged).
- **Per-collection settings** (`rag.py`, `config.py`): TOP_K, MIN_SCORE,
  MIN_COVERAGE, MIN_DENSE_SIM, DENSE_SOLO_SIM, MAX_PER_WORK, POOL_FACTOR,
  HIERARCHICAL, RERANK, TEMPERATURE and the quotes settings are read through
  `col.setting()`, so `settings.json` overrides apply per collection.
- **Numbered citations** (`rag.py`, `serve.py`, `ask.py`,
  `conversations.py`): excerpts are numbered `[1]…[n]`; scholar/persona
  prompts ask for `[n]`; `check_citations()` reports used/unused/out-of-range
  markers; `/ask` emits a `citations` event and saves it with the turn.
  Quotes mode unchanged.
- **Follow-ups** (`rag.py`, `serve.py`, `ask.py --chat`): the last 3 turns
  are loaded; a deterministic follow-up detector triggers one standalone
  rewrite call used for retrieval and the gate; history goes into the prompt
  inside the token budget and is trimmed before excerpts; `meta` reports
  `rewritten_question`.
- **UI** (`static/index.html`): `[n]` markers link to the reader, sources
  list numbered with uncited ones dimmed, warning chip for out-of-range
  markers, "searched for: …" line; reopened conversations render the same.
- Tests: `tests/test_m3a_answer_quality.py` (15); `tests/conftest.py`
  fake LLM accepts a callable reply; baseline
  `tests/fixtures/baselines/M3.json` (177 passed).

## 2026-11-11

### Bug fix — M2: dense retrieval on the ONNX path, relevance gate, eval tooling
Milestone M2 of `implem-plan-enhancements.md` (P0-1…P0-3, P1-1), run as two
parallel subagents plus one follow-up. `bash scripts/verify.sh M2` passes;
measured numbers are in `plan.md` ("M2 eval").
- **Embedding actually runs** (`encoder.py` new; `ingest.py`, `rag.py`,
  `serve.py`, `scripts/export_onnx.py`): one shared encoder (bundled ONNX,
  else sentence-transformers) for ingest, queries and imports. Before this,
  linked collections were never embedded without sentence-transformers.
- **Encoder matches the model**: CLS pooling and a 512-token limit read from
  the model's own config files (was mean pooling at 256 tokens). Vectors are
  tagged with pooling/max_len and re-embed when they don't match; encoding is
  batched with per-batch progress events.
- **Relevance gate** (`rag.py`, `config.py`): the dense branch accepted
  every question once vectors existed (cosine ≥ 0.25 OR'd with BM25). Now
  raw cosine with a tiered rule — accept at cosine ≥ `DENSE_SOLO_SIM` (0.72),
  or when BM25 passes and cosine ≥ `MIN_DENSE_SIM` (0.57). 0 false accepts
  / 0 false rejects on 150 labelled fixture+IBON questions (old rule: 95
  false accepts). BM25-only collections unchanged.
- **`eval.py`** rewritten: hit@1/3/k, recall@k, MRR, gate false-accepts and
  false-rejects, `--json`, `--variants` (bm25 / hybrid / hierarchical /
  +rerank).
- Measured for later: 8 of 14 IBON chunks exceed the encoder's 512-token
  limit (`CHUNK_WORDS` unchanged for now).
- Tests: `tests/test_m2a_encoder.py`, `test_eval.py`,
  `test_m2c_dense_gate.py`; `test_gate.py` expects the ice-cores false
  reject only without vectors; new baseline `tests/fixtures/baselines/M2.json`
  (162 passed).

### Bug fix — M1: index hygiene, server robustness, local-server hardening
Milestone M1 of `implem-plan-enhancements.md` (P0-4…P0-9, P1-3 in
`plan-enhancements.md`), run as three parallel subagents plus one follow-up.
`bash scripts/verify.sh M1` passes; retrieval numbers unchanged vs M0.
- **Ingestion** (`ingest.py`, `library.py`, `summarize.py`): derived data for
  deleted sources is pruned; sections under 40 words and short trailing
  windows are no longer dropped; repeated section titles stay separate
  (`X (2)`); re-chunking is decided by source sha256 + chunk settings, not
  mtime; registry, `collection.json` and manifest are written atomically
  and carry a schema version.
- **Same-stem sources** (P0-5; `library.py`, `ingest.py`, `rag.py`,
  `dates.py`, `quotes_bank.py`, `summarize.py`, `serve.py`): derived files
  are keyed by full filename (`report.md.jsonl`), so `report.md` and
  `report.txt` no longer overwrite each other; chunk ids are unique across a
  collection and unchanged for sources without a clash. Existing collections
  migrate to schema 2 on next ingest by renaming files (vectors and
  summaries stay valid; rehearsed on a copy of the IBON collection: 36 files
  renamed, 0 deleted).
- **Generation** (`rag.py`): prompts are budgeted to `LLM_CTX` (lowest-ranked
  excerpts dropped, `dropped` reported in the `/ask` meta event);
  `max_tokens` sent to the openai and embedded backends; embedded model
  serialized with a lock; `NotIndexed` replaces `SystemExit`; AskJ-only era
  pattern removed.
- **Server** (`serve.py`, `app.py`, `conversations.py`): un-indexed
  collections return 409 instead of dropping the connection; true LRU
  runtime cache with loads outside the global lock; Host/Origin allow-list,
  JSON content-type requirement, 10 MB JSON body cap; desktop app uses a
  per-launch token exchanged for a SameSite=Strict cookie. Neutral persona
  disclaimer in exports; neutral test string in `scripts/export_onnx.py`.
- CLIs (`ask.py`, `eval.py`) exit with the "run ingest" message on
  un-indexed collections.
- Tests: strict-xfail markers removed for all fixed defects; new
  `tests/test_m1a_ingest.py`, `test_m1b_server.py`, `test_m1c_generation.py`,
  `test_m1d_artifact_keys.py` (127 passed, 1 xfail left for M2).

### Feature — Test harness and milestone verification (M0)
Built milestone M0 of `implem-plan-enhancements.md` so later enhancement
milestones can be run with `/goal` and checked mechanically. No app code
changed.
- pytest suite on a temporary `ANYRAG_HOME` (never touches `devdata/`),
  with a fake OpenAI-compatible LLM and an in-process server fixture
  (`tests/conftest.py`, `tests/helpers.py`, `pytest.ini`,
  `requirements-dev.txt`).
- Generated fixture corpus covering md/txt/html/docx/pdf plus the deliberate
  defect cases (`tests/fixtures/build.py`, `tests/fixtures/corpus/`), and 20
  eval questions (`tests/fixtures/eval.jsonl`).
- 70 baseline tests: registry/isolation, ingest per format, reader round
  trip, quote verification, relevance gate, HTTP end to end.
- 20 strict-xfail tests pinning known defects for M1/M2
  (`tests/test_known_defects.py`), including a newly found one: the last
  chunk of a section is dropped when `CHUNK_OVERLAP_WORDS` < 40.
- `scripts/verify.sh` + `scripts/verify_checks.py`: per-milestone gate,
  fixture eval yardstick (`tests/eval_fixture.py`) with regression baselines
  (`tests/fixtures/baselines/M0.json`), and a check that `devdata/` and
  `writings/` are unchanged.
- `errors.py` (`NotIndexed`, for M1); `.gitignore` ignores `.pytest_cache/`;
  `CLAUDE.md` documents the test commands; `implem-plan-enhancements.md`
  updated with the as-built harness.

### Feature — Agent implementation plan
Added `implem-plan-enhancements.md`: the enhancements split into milestones
M0–M5 for running with `/goal` and subagents, with per-agent file
ownership, shared rules, verification commands and paste-ready goal text.
Requested so the enhancement work can be executed by agents.

### Feature — Enhancement review
Added `plan-enhancements.md`: a review of the current state and prioritized
enhancements (P0 correctness bugs through P4 packaging) with a suggested
order. Requested as a review of the project's current state.
