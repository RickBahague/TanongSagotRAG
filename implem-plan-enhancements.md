# AnyRAG — Implementation Plan for `plan-enhancements.md`

Execution plan for running the enhancements with `/goal` and subagents.
Item ids (P0-1, P2-8, …) refer to `plan-enhancements.md`; read that file for
the reasoning behind each item. This file says who does what, in what order,
and how "done" is checked.

## How to run it

One `/goal` per milestone, in order. Each milestone ends with a single
command whose output decides completion:

```bash
bash scripts/verify.sh M1      # prints "VERIFY M1: PASS" or "VERIFY M1: FAIL" + reasons
```

Paste the goal text given under each milestone. Between milestones, review
the diff and commit it yourself (agents never commit), so each milestone
starts from a clean tree and can be rolled back on its own.

| Milestone | Covers | Agents |
|---|---|---|
| M0 | Test harness, fixtures, verify script | 1, serial |
| M1 | Index hygiene, server robustness, hardening (P0-4…P0-9, P1-3) | 3 parallel, then 1 |
| M2 | Dense retrieval on the ONNX path, gate, eval tooling (P0-1…P0-3, P1-1) | 2 parallel, then 1 |
| M3 | Follow-ups, numbered citations, per-collection tuning (P2-1, P2-2, P2-8) | 1, then 1 |
| M4 | Settings, source management, ingest and chat UX (P3-1, P3-2, P3-4, P3-5) | 1, then 1 |
| M5 | Remaining retrieval items, one goal each (P2-3…P2-7, P2-9) | 1 per item |

Not in any agent run (needs you): macOS signing/notarization (needs an
Apple Developer account), pushing so the CI workflow runs on GitHub for the
first time, and a click-through of the UI by hand (there are no browser
tests; the server flows are driven over HTTP, `plan.md` "P4 items"). Done
on 2026-11-19: the real-collection baseline (`jms`, 96 questions), CI
workflow, first-run model download, log file, app icon, GPU layers, the
ONNX reranker (M5-6) and the chunk-size and header-stripping tests.

## Rules every agent prompt must include

1. Python only through `.venv/bin/python`. Never system Python.
2. Never read-write `devdata/` or `writings/`. Tests and experiments use a
   temporary `ANYRAG_HOME`. `writings/IBON` may be *linked* read-only from a
   scratch home.
3. No `git add`, `git commit`, `git push`, no branch or worktree changes.
   All agents share one working tree.
4. Edit only the files listed as yours for the task. If a fix needs a file
   you don't own, stop and report it to the orchestrator instead of editing.
5. No new runtime dependencies. `pytest` (dev only) is the one addition.
   Do not install `sentence-transformers`/torch.
6. No network access except the one fetch named in M2-A.
7. A task is done when its listed tests pass and the whole suite still
   passes: `.venv/bin/python -m pytest -q`.
8. Any change to retrieval scoring or gating must report before/after
   numbers from `eval.py`; the orchestrator records them in `plan.md`.
9. Report format: files changed, tests added/unskipped, the pytest summary
   line, and anything left undone. Do not claim a check you did not run.

Orchestrator duties: launch the agents for a wave in one message, wait for
all of them, run `bash scripts/verify.sh <M>`, and send failures back to the
owning agent. Do not fix another agent's files inline while it is running.

## Known trap for tests

`config.USER_ROOT`, `library.REGISTRY` and `library.COLLECTIONS_DIR` are
computed at import time. A fixture that only sets `ANYRAG_HOME` after import
will write into `devdata/`. M0 must solve this once (set the env var in
`conftest.py` before any project import, and assert in a session fixture
that `config.USER_ROOT` is under the temp dir), and every later test relies
on it.

---

## M0 — Harness

One agent, serial. Nothing else can be verified until this exists.

**Create**
- `requirements-dev.txt` (`pytest`), installed into `.venv`.
- `errors.py`: `class NotIndexed(Exception)` (shared by M1-B and M1-C).
- `tests/conftest.py`: temp `ANYRAG_HOME` (see trap above); `make_collection`
  helper; `fake_llm` fixture — a stdlib OpenAI-compatible stub server that
  streams a canned reply and records the request body; `server` fixture
  wrapping `serve.make_server(0)` in a thread.
- `tests/fixtures/corpus/`: small md, txt, html, docx and PDF files on
  clearly distinct topics, written by the agent (generate the docx/PDF with
  a script in `tests/fixtures/build.py`; pypdf can write a text PDF). Include
  a 20-word txt, two files sharing a stem (`report.md`, `report.txt`), and a
  PDF with two sections of the same title.
- `tests/fixtures/eval.jsonl`: 15 on-topic questions with `expect` regexes
  and 5 off-topic ones with `"expect": null`.
- Baseline tests that pass on today's code: registry CRUD and isolation,
  ingest of every format, `passage()` round trip, `verify_quotes`
  accept/reject, gate on the fixture corpus, HTTP smoke (create → ingest
  stream → stats → ask with `fake_llm` → conversation saved).
- One `xfail(strict=True)` test per defect below, named as shown. Later
  milestones remove the marker when they fix the bug.

| Test | Item |
|---|---|
| `test_prune_removed_source` | P0-4 |
| `test_same_stem_sources_both_indexed` | P0-5 |
| `test_context_fits_budget`, `test_max_tokens_sent_openai` | P0-6 |
| `test_embedded_generation_serialized` | P0-7 |
| `test_unindexed_collection_returns_409` | P0-8 |
| `test_short_document_is_chunked`, `test_chunk_tail_not_dropped`, `test_duplicate_section_titles_kept`, `test_rechunk_on_content_change_not_mtime`, `test_runtime_cache_is_lru`, `test_json_writes_atomic`, `test_no_askj_leftovers` | P0-9 |
| `test_foreign_host_rejected`, `test_foreign_origin_rejected`, `test_post_requires_json_content_type`, `test_oversized_body_rejected` | P1-3 |
| `test_embed_stage_uses_onnx_encoder` | P0-1 |

- `scripts/verify.sh <M>`: runs pytest, then the per-milestone checks listed
  under each milestone below, and prints exactly one final line
  `VERIFY <M>: PASS` or `VERIFY <M>: FAIL`. For M1+ it fails if any test
  listed for that milestone is still marked xfail or skip.

*As built (M0 done):* defect tests live in `tests/test_known_defects.py`
(`xfail_strict` is on, so a fix XPASSes and fails the suite until its
marker is removed). `verify.sh` runs pytest, then `tests/eval_fixture.py`
(fixed yardstick: hit@1/3/10, MRR, gate false-accepts/rejects on the clean
fixture files, independent of `eval.py`), then `scripts/verify_checks.py`
(required-test lists + per-milestone checks), then confirms `devdata/` and
`writings/` are byte-for-byte unchanged. On PASS of M0/M2/M3/M4 it writes
`tests/fixtures/baselines/<M>.json`; M1–M4 fail if hit@10, MRR or gate
false-accepts regress against the latest earlier baseline. M0 baseline:
BM25, hit@1 0.933, hit@10 1.0, MRR 0.967, 0 false-accepts, 1 false-reject
("ice cores … past climate", pinned in `tests/test_gate.py`).

Added during M0: `test_chunk_tail_not_dropped` (P0-9, owned by M1-A). The
final window of a section is dropped when it is < 40 words, which happens
only when `CHUNK_OVERLAP_WORDS` < 40 (a per-collection override); defaults
(80) never hit it.

**Done when** `bash scripts/verify.sh M0` prints `VERIFY M0: PASS` (suite
green, the xfail tests above all present and failing as expected, nothing
written under `devdata/`).

**Goal text**
> Complete milestone M0 of implem-plan-enhancements.md. Done when
> `bash scripts/verify.sh M0` prints `VERIFY M0: PASS` and `git status`
> shows no changes under devdata/ or writings/. Do not commit.

---

## M1 — Index hygiene, robustness, hardening

### Wave 1 — three agents in parallel (disjoint files)

**M1-A · ingestion** — owns `ingest.py`, `library.py`, `dates.py`,
`quotes_bank.py`, `summarize.py`
- P0-4: prune chunks/paras/dates/quotes/summaries/vectors and manifest
  entries for sources no longer present; emit a `prune` progress event.
- P0-9: keep the first chunk of a section even under 40 words; keep
  duplicate section titles distinct (suffix ` (2)`); replace the mtime check
  in `chunk_source` with a source sha256 stored in the manifest; atomic JSON
  writes via one helper (temp file + `os.replace`) used for the registry,
  `collection.json` and the manifest; add `"schema": 1` to both.
- Unskips: `test_prune_removed_source`, `test_short_document_is_chunked`,
  `test_chunk_tail_not_dropped`, `test_duplicate_section_titles_kept`,
  `test_rechunk_on_content_change_not_mtime`, `test_json_writes_atomic`.

**M1-B · server** — owns `serve.py`, `conversations.py`, `app.py`,
`scripts/export_onnx.py`
- P0-8: catch `errors.NotIndexed` and return 409 JSON in GET and POST paths
  (M1-C raises it; until then also treat `SystemExit` from `runtime()` the
  same way so this agent's tests pass alone).
- P0-9: true LRU for `_RUNTIMES`; build the `Runtime` outside
  `_RUNTIME_LOCK`; remove the AskJ leftovers in `conversations.py` and
  `export_onnx.py`.
- P1-3: `Host` allow-list (`127.0.0.1`/`localhost` on the bound port),
  `Origin` check when the header is present, `Content-Type:
  application/json` required on JSON POSTs, body cap in `_body()`,
  per-launch token set by `app.py` and required when configured.
- Unskips: `test_unindexed_collection_returns_409`,
  `test_runtime_cache_is_lru`, the four P1-3 tests, and the
  conversations/export half of `test_no_askj_leftovers`.

**M1-C · generation** — owns `rag.py`
- P0-6: token-budget `build_context()` to
  `LLM_CTX − LLM_MAX_TOKENS − overhead` (estimate tokens as
  `words × 1.35`), dropping lowest-ranked hits; `ask_stream` returns
  `dropped` in its result; send `max_tokens` on the `openai` and `embedded`
  backends.
- P0-7: a module-level lock around embedded generation, held for the whole
  stream.
- P0-8: `load_sources` raises `errors.NotIndexed`; `ask.py`, `ingest.py`
  and `eval.py` CLIs convert it to the existing exit message (orchestrator
  applies the one-line CLI edits after the wave if M1-C cannot touch them).
- P0-9: delete the `during martial law` era pattern.
- Unskips: `test_context_fits_budget`, `test_max_tokens_sent_openai`,
  `test_embedded_generation_serialized`, the `rag.py` half of
  `test_no_askj_leftovers`.

### Wave 2 — one agent, after wave 1 is green

**M1-D · artifact keying (P0-5)** — owns every file that maps a source to
its derived files: `library.py`, `ingest.py`, `rag.py`, `dates.py`,
`quotes_bank.py`, `summarize.py`, `serve.py`
- Add `Collection.artifact_key(filename)` (full filename, sanitized; not the
  stem) and use it everywhere a derived path is built or parsed. Grep for
  `.stem` and justify each remaining use in the report.
- Migration: on ingest, a collection with no `schema` or `schema: 1` and
  stem-keyed files is re-keyed by renaming, without re-chunking; bump to
  `schema: 2`.
- Unskips: `test_same_stem_sources_both_indexed`; adds
  `test_stem_keyed_collection_migrates`.

**verify.sh M1 additionally checks:** no xfail/skip left on the M1 tests; a
scratch-home ingest of `writings/IBON` (linked) produces the same chunk
count as before the milestone (14) and one sample question passes the gate;
fixture eval not below the M0 baseline.

**Goal text**
> Complete milestone M1 of implem-plan-enhancements.md using parallel
> subagents for wave 1 (M1-A, M1-B, M1-C) and one subagent for wave 2
> (M1-D), following the file-ownership rules in that document. Done when
> `bash scripts/verify.sh M1` prints `VERIFY M1: PASS`. Do not commit; do
> not modify devdata/ or writings/.

---

## M2 — Dense retrieval on the ONNX path

### Wave 1 — two agents in parallel

**M2-A · encoder (P0-1, P0-2)** — owns new `encoder.py`, plus `ingest.py`,
`rag.py`, `serve.py`
- `encoder.get_encoder()`: ONNX when `models/encoder-onnx/model.onnx`
  exists, else sentence-transformers, else `ImportError`. One cached
  instance. Used by `embed_source`, `DenseIndex`, `SummaryIndex`.
  `import_work` calls `embed_source` instead of its own copy.
- Pooling: fetch `1_Pooling/config.json` and
  `sentence_bert_config.json` from the `EMBED_MODEL` repo on Hugging Face
  (the one permitted network call; two small JSON files) and make
  `OnnxEncoder` match them (pooling mode, max sequence length). Have
  `scripts/export_onnx.py` download both files next to `model.onnx` so the
  encoder reads its configuration from disk from then on. If the fetch
  fails, stop and report; do not guess.
- Vector files gain `pooling` and `max_len` tags; `vectors_stale` and
  `DenseIndex` treat a mismatch as stale.
- Batch the ONNX encode (`EMBED_BATCH`) and emit embed progress events.
- Report how many fixture and IBON chunks exceed the token limit, so the
  `CHUNK_WORDS` question in P0-2 can be decided with a number.
- Unskips `test_embed_stage_uses_onnx_encoder`; adds
  `test_vectors_stale_on_pooling_change`, `test_hybrid_index_loads`.
  Tests needing the model file are skipped with a clear reason when
  `models/encoder-onnx/` is absent (it is present on this machine).

**M2-B · eval tooling (P1-1)** — owns `eval.py`
- hit@k, recall@k and MRR; `"expect": null` questions scored as gate
  false-accepts; on-topic questions blocked by the gate scored as
  false-rejects; `--json`; `--variants` to compare bm25 / hybrid /
  hierarchical / +rerank where available.
- Adds `tests/test_eval.py` on the fixture corpus with the BM25 path.

### Wave 2 — one agent

**M2-C · dense gate (P0-3)** — owns `rag.py`, `config.py`
- In a scratch home: ingest the fixture corpus and linked `writings/IBON`
  with vectors; dump top-1 cosine for every eval question (on- and
  off-topic).
- Add `MIN_DENSE_SIM` with its own threshold chosen from that data; stop
  reusing `MIN_SCORE` for dense and remove the `* 20` scaling if nothing
  else needs it. Decide OR vs AND for the two signals from the measured
  false-accept/false-reject rates and state the numbers.
- Adds `test_dense_gate_rejects_off_topic` (model-dependent, skippable).
- Orchestrator records BM25 vs hybrid vs hierarchical eval numbers and the
  chosen threshold in `plan.md`, labelled as fixture + IBON (small corpora;
  to be re-run on a real collection).

**verify.sh M2 additionally checks:** a scratch-home ingest of the fixture
corpus produces `.npz` files with `model`, `pooling`, `max_len`, `src_hash`;
`load_index` returns a hybrid-backed index; `tests/eval_fixture.py` reports
hit@10 and MRR ≥ the M0 baseline and gate false-accepts ≤ it; `plan.md`
contains an "M2 eval" section. (`eval.py --json` is M2-B's deliverable for
real collections; verification uses the fixed fixture yardstick.)

**Goal text**
> Complete milestone M2 of implem-plan-enhancements.md: wave 1 with two
> parallel subagents (M2-A, M2-B), then M2-C. Done when
> `bash scripts/verify.sh M2` prints `VERIFY M2: PASS` and plan.md has an
> "M2 eval" section with the measured numbers. Do not commit; do not
> install sentence-transformers.

---

## M3 — Answer quality

Serial: `rag.py` is the hot file, so one backend agent, then one UI agent.

**M3-A · backend** — owns `rag.py`, `serve.py`, `ask.py`, `config.py`,
`conversations.py`
- P2-8: route `TOP_K`, `MIN_SCORE`, `MIN_COVERAGE`, `MIN_DENSE_SIM`,
  `DENSE_SOLO_SIM` (added in M2-C),
  `MAX_PER_WORK`, `POOL_FACTOR`, `HIERARCHICAL`, `RERANK`, `TEMPERATURE`
  and the quotes settings through `col.setting()`. Defaults unchanged.
  Test: two collections with different overrides behave differently in one
  process.
- P2-2: number excerpts `[1]…[n]` in `build_context`; scholar and persona
  prompts ask for `[n]` markers; new `check_citations(answer, n)` returns
  used, unused and out-of-range markers; `/ask` emits a `citations` event
  and stores it in the turn. Quotes mode keeps its existing format and
  verification.
- P2-1: `/ask` loads the last 3 turns of `conversation_id`. A follow-up
  detector (pronoun/ellipsis/short-question heuristic) triggers one
  standalone-question rewrite call; retrieval and the gate use the rewrite;
  recent turns go into the prompt inside the M1 token budget. `meta`
  reports `rewritten_question`. `ask.py --chat` does the same in memory.
- Tests with `fake_llm`: rewrite is called only for follow-ups; the
  rewritten text is what gets retrieved on; history is trimmed before
  excerpts are; out-of-range `[n]` is flagged; first-turn behaviour and the
  NDJSON event order are unchanged apart from the new events.

**M3-B · UI** — owns `static/index.html`
- Render `[n]` markers as links that open the reader on source n; dim
  unused sources; show a warning chip for out-of-range markers; show
  "searched for: …" when `rewritten_question` is present; reopen saved
  conversations with the same rendering.

**verify.sh M3 additionally checks:** eval numbers on the fixture
collection are not lower than the M2 numbers (P2-8 must not change
defaults); `node --check` on the extracted inline script if `node` is
available, otherwise a note that the UI was not machine-checked.

**Manual check for you after M3:** ask a question, ask a follow-up with a
pronoun, click an inline `[n]`.

**Goal text**
> Complete milestone M3 of implem-plan-enhancements.md: subagent M3-A,
> then subagent M3-B. Done when `bash scripts/verify.sh M3` prints
> `VERIFY M3: PASS`. Do not commit. End with the list of UI behaviours
> that were not machine-verified.

---

## M4 — Product

**M4-A · backend** — owns `serve.py`, `library.py`, `ingest.py`,
`config.py`, `rag.py`
- P3-1: `POST /settings` and `GET|POST /c/<id>/settings`, validated against
  an allow-list with types and ranges; applied without restart (runtimes
  reloaded; `config` values re-read through one accessor);
  `GET /backend/status` reporting reachability, model presence and, for
  Ollama, the model list from `/api/tags`.
- P3-2: `PATCH /c/<id>` (name, description, author, persona);
  `DELETE /c/<id>/source/<name>` (managed: delete file and artifacts;
  linked: refuse with a clear message); `POST /c/<id>/ingest?source=<name>`;
  recursive scan of linked folders with ignore patterns, keyed by relative
  path through `artifact_key`; manifest `error` and a `no_text` flag for
  text-less PDFs exposed in `/stats`; on-disk size in `/collections`.
- P3-4: ingest runs as a background job with a `done`/`total` count per
  stage; `GET /c/<id>/ingest/status`; `POST /c/<id>/ingest/cancel`; the
  NDJSON stream becomes a view on the job so a second client can attach.
- Tests for every route, including validation failures, the linked-source
  delete refusal, cancel, and re-attach.

**M4-B · UI** — owns `static/index.html`, `app.py`
- Settings panel with "Test connection" and the Ollama model dropdown;
  edit-collection form; remove and re-ingest buttons, error and "no text"
  badges in the source table; delete confirmation showing size.
- Progress bar driven by `done`/`total`; cancel button.
- P3-5: Markdown rendering for scholar/persona answers (small inline
  renderer, text nodes only, no HTML passthrough); Stop button via
  `AbortController`; copy answer.
- P3-3: `js_api` folder picker in `app.py`, used by the wizard when
  `window.pywebview` exists; typed path stays as the fallback.

**Manual check for you after M4:** settings round trip, remove a source,
cancel an ingest, stop a streaming answer, pick a folder in the desktop
window.

**Goal text**
> Complete milestone M4 of implem-plan-enhancements.md: subagent M4-A,
> then subagent M4-B. Done when `bash scripts/verify.sh M4` prints
> `VERIFY M4: PASS`. Do not commit. End with the list of UI behaviours
> that were not machine-verified.

---

## M5 — Retrieval items, one goal each

Each of these changes scoring or chunking, so each is its own goal, and each
is kept only if it meets its "keep if" rule.

| Goal | Item | Owns | Keep if |
|---|---|---|---|
| M5-1 | P2-5 inverted-index BM25, gate reuses hits, token cache | `rag.py` | identical rankings on fixtures; faster on a synthetic 50k-chunk corpus |
| M5-2 | P2-9 idf-weighted coverage, "closest material", search-only mode | `rag.py`, `serve.py`, `static/index.html` | gate false-accepts down, false-rejects not up |
| M5-3 | P2-3 heading segmentation for PDF/HTML/TXT/DOCX | `ingest.py` | fewer sources on rung `none`; hit@k not lower |
| M5-4 | P2-4 paragraph-aware chunking with explicit offsets | `ingest.py`, `rag.py` | `passage()` round trip still exact; hit@k not lower |
| M5-5 | P2-6 Unicode tokenizer, per-collection stopwords | `rag.py` | English numbers unchanged; non-ASCII fixture passes |
| M5-6 | P2-7 ONNX reranker | `rag.py`, `encoder.py`, `scripts/` | MRR up on the real collection (needs a model download: ask first) |

**How an M5 goal is run** (method, not part of the goal condition)
- Use one subagent per item. If that agent is stopped or fails, resume it,
  or let the orchestrator finish the item and say so in `plan.md`. Who did
  the work is not a completion check.
- Before editing, snapshot the files the item owns outside the repo. A
  revert restores that snapshot; never `git checkout`, which would also
  wipe earlier uncommitted items.
- Measure before and after on the fixtures (`tests/eval_fixture.py`) and on
  IBON in a scratch home. If `EVAL_COLLECTION` names a real collection,
  measure there too with `eval.py --json`. `verify.sh` does not read
  `EVAL_COLLECTION`; it checks the suite, the fixture eval, that `plan.md`
  has an `M5-<n>` entry, and that `devdata/` and `writings/` are unchanged.
- The orchestrator appends a `## M5-<n> eval` section to `plan.md` with the
  before/after numbers and the verdict: kept, or reverted and why.

**Goal texts** (paste one; each names its item, and its condition is only
what `verify.sh` and `plan.md` can show)

> Complete M5-1 of implem-plan-enhancements.md. Done when
> `bash scripts/verify.sh M5-1` prints `VERIFY M5-1: PASS` and plan.md has
> an "M5-1 eval" section with before/after numbers and a verdict (kept, or
> reverted because the keep-if rule was not met). Do not commit.

Replace `M5-1` with `M5-2` … `M5-5` for the other items. Do not paste the
text with a placeholder such as `M5-<n>`: the goal checker cannot tell which
item was meant and will never mark it complete.

> Complete M5-6 of implem-plan-enhancements.md. Done when
> `bash scripts/verify.sh M5-6` prints `VERIFY M5-6: PASS` and plan.md has
> an "M5-6 eval" section with before/after MRR on collection <id> and a
> verdict. You may download the reranker model named in that section. Do
> not commit.

M5-6 needs two things from you first: approval of the model download (put
it in the goal text, as above) and a real multi-document collection with an
`eval.jsonl`, whose id replaces `<id>`.

**Real collection (2026-11-19):** `jms` (12 books, 4,257 chunks) now has a
35-question `eval.jsonl`; use it as `<id>` and as `EVAL_COLLECTION`. Its
first numbers are in `plan.md` ("JMS eval").

**Why the old template failed as a goal** (2026-11-12)
- It was pasted with `M5-<n>` unfilled, so no item was ever named.
- "with one subagent" was part of the condition. It describes how the work
  is done, cannot be checked from any output, and can never be satisfied
  again once an agent is stopped partway.
- "revert … and stop" made a reverted item look unfinished. A recorded
  revert is now a valid way to complete the goal.
- It told you to run M5 only with a real collection and `EVAL_COLLECTION`,
  but `verify.sh` never used that variable.

**As run (2026-11-12, uncommitted at the time of writing)** — fixtures and
IBON only; re-measure on a real collection. Numbers are in `plan.md`.

| Goal | Result |
|---|---|
| M5-1 | kept: identical rankings, search about 110× faster at 50k chunks, load time about the same |
| M5-2 | idf-weighted gate reverted (false rejects went up); closest material and search-only kept |
| M5-3 | kept: sources on rung `none` 10 → 8, hit@k unchanged |
| M5-4 | kept: round trip exact, hit@k unchanged |
| M5-5 | kept: English results identical, non-ASCII test passes; off-topic Greek still passes the hybrid gate (strict xfail) |
| M5-6 | not run: needs the model download and a real collection |

---

## Deferred (not planned for agents)

P3-6 extra formats and OCR, P3-7 cross-collection search, and from P4:
signing/notarization, schema version, CLI subcommands, doc refresh. (P4's
CI matrix, first-run model download, icon, GPU layers and logging were done
on 2026-11-19.) Open from measurement: the multilingual `EMBED_MODEL`
(needs a download) and the two hard JMS negatives the gate still accepts
(function-word stopwords and BM25-only test coverage were done on
2026-11-19). Pick these up as separate goals once M1–M4 are
merged; the doc refresh in P4 is small enough to do by hand at that point.
