# Changelog

## 2026-11-11

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
