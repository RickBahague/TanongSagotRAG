# Changelog

## 2026-11-11

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
