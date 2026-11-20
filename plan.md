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

## M2 eval — 2026-11-11 (fixture + IBON; small corpora, re-run on a real collection)

Dense retrieval now runs on the bundled ONNX encoder (`encoder.py`):
bge-small-en-v1.5 with **CLS pooling, 512-token limit**, read from the model's
own `1_Pooling/config.json` / `sentence_bert_config.json` (previously the
encoder mean-pooled and truncated at 256 — P0-2). Vectors carry
`model/pooling/max_len/src_hash` tags; older untagged vectors re-embed once.
ONNX vs sentence-transformers parity is still unverified (sentence-transformers
not installed by design).

**Chunks over the 512-token encoder limit** (input to the P0-2 `CHUNK_WORDS`
decision, not yet changed): fixture 0/16 (max 170 tokens); writings/IBON
**8/14** (median 607, max 676 tokens) — the tail of most full-size IBON chunks
is not embedded.

**Dense gate (P0-3).** The old dense branch accepted at cosine ≥ 0.25
(`cos*20 >= MIN_SCORE`) and was OR'd with BM25, so once vectors existed every
question passed. Measured raw top-1 cosine over 135 labelled pairs (fixture
corpus, 5 single-topic fixture collections asked the other topics, IBON with
10 on-topic + 5 off-topic questions): on-topic 0.61–0.85, off-topic 0.29–0.64 —
the groups overlap, so no single threshold separates them.

| Gate rule (FA / FR over 95 neg / 40 pos) | FA | FR |
|---|---|---|
| old: BM25 OR cos ≥ 0.25 | 95 | 0 |
| BM25 only | 1 | 2 |
| dense only, best (t = 0.61) | 1 | 0 |
| BM25 OR dense ≥ 0.64 | 1 | 0 |
| BM25 AND dense ≥ 0.57–0.61 | 0 | 2 |
| **chosen: cos ≥ 0.72, or (BM25 ok AND cos ≥ 0.57)** | **0** | **0** |

Also 0/0 on 15 extra cross-corpus negatives (fixture questions asked of IBON).
New settings `MIN_DENSE_SIM = 0.57` (margin ≈0.04 each side) and
`DENSE_SOLO_SIM = 0.72` (margin ≈0.08 each side); BM25-only collections are
unchanged. Known limit: paraphrased on-topic questions with little word
overlap and cosine < 0.72 will be gated out.

**Retrieval before → after the gate change** (`eval.py --json --variants`;
hierarchical and +rerank unavailable — no summaries, no reranker):

| Collection / variant | hit@1 | hit@10 | recall@10 | MRR | FA | FR |
|---|---|---|---|---|---|---|
| fixture, bm25 | 0.933 | 1.0 | 0.967 | 0.967 | 0 → 0 | 1 → 1 |
| fixture, hybrid | 0.933 | 1.0 | 1.0 | 0.967 | 5 → 0 | 0 → 0 |
| IBON, bm25 | 0.90 | 1.0 | 0.967 | 0.933 | 0 → 0 | 0 → 0 |
| IBON, hybrid | 0.90 | 1.0 | 0.967 | 0.933 | 5 → 0 | 0 → 0 |

Hybrid equals BM25 on hit@k/MRR here (corpora too small to discriminate);
it improves recall@10 on the fixture and fixes the ice-cores false reject
through the dense gate. Verify yardstick (`tests/eval_fixture.py`, HybridIndex):
hit@1 0.933, hit@10 1.0, MRR 0.967, FA 0, FR 0.

## M5-1 eval — 2026-11-12 (P2-5: inverted-index BM25, gate reuses hits, token cache)

Keep-if rule: identical rankings on the fixtures, faster on a synthetic
50k-chunk corpus. **Kept.**

- `BM25Index` keeps an inverted index (term → postings `(doc, tf)`) and
  scores only the docs that hold a query term. It uses the same float
  expression in the same query-token order as the old full scan, and ties
  still break on `(score, idx)`, so scores and rankings are identical.
  Per-query scores are memoized, so the gate and the retrieval that follows
  it score a question once. `gate_ok`/`bm25_gate_ok` also accept the
  caller's hits (`bm25_hits=` / `hits=`) instead of searching again.
- Token cache: `data/cache/tokens/<artifact key>.json`, one file per chunk
  file, holding `tokenize(text)` per chunk. It is keyed by sha256 of the
  chunk texts plus `TOKENIZER_VERSION` and a stopword-list fingerprint.
  Section titles are always re-tokenized (they can change without the chunk
  file changing). A corrupt or mismatched cache is rebuilt silently, and
  files for removed sources are deleted when the index loads.

**Fixture eval** (`tests/eval_fixture.py`, before → after): output JSON
byte-identical — HybridIndex, hit@1 0.933, hit@3 1.0, hit@10 1.0, MRR 0.967,
gate false-accepts 0, false-rejects 0. Test check against a verbatim copy of
the old scorer (`tests/test_m5_1_bm25.py`): identical (score, id) lists on
every fixture eval question at k = 1/3/10/all, and on a 650-chunk random
Zipfian corpus with forced ties (150 queries × 4 k values).

**Synthetic 50k-chunk benchmark** (stdlib only, 100–200 words per chunk,
Zipfian vocabulary of 30k words, seed 42, 200 queries of 2–7 terms; script
kept outside the repo):

| | old (full scan) | new (inverted index) |
|---|---|---|
| top-10 rank mismatches (200 queries) | — | **0** |
| search k=10, median / p95 | 28.4 / 39.8 ms | **0.29 / 1.77 ms** |
| gate + search (ask path), median / p95 | 60.1 / 80.1 ms | **0.51 / 2.91 ms** |
| index load, warm token cache | 2.96 s (build) | **2.16 s** (1.67 s with GC off) |
| index load, cold (tokenize + write cache) | 2.81 s | 4.36 s, once per changed chunk file |

The cache saves tokenization (≈1.9 s of the old build), but building the
postings lists then takes most of the load (≈1.1–1.8 s). Next step if load
time matters: cache the postings themselves, or build them with GC paused.
The cold path is slower once, because it also writes the cache. All numbers
are from one run on this machine.

**Independent re-run** (M5-1 subagent review, same scripts): 0 rank
mismatches; search 28.2 → 0.34 ms median; gate + search 58.0 → 0.49 ms;
old build 2.62 s vs new warm-cache load 2.52 s (bench script), or 1.64 s
when the load is measured on its own (prof2). The search and gate speedups
reproduce (≈80–120×). The load-time gain varies between runs and is small.
The review also fixed `search_idx` for k ≤ 0 to match the old slice
semantics; no caller passes it.

**MRR on collections, before → after** (added 2026-11-13; scratch home,
sources linked read-only; "before" = the pre-M5-1 full-scan scorer run on
the same chunks, "after" = `BM25Index`; k = 10):

| Collection | Chunks | Questions | BM25 MRR before → after | hit@1 before → after | Top-10 lists identical |
|---|---|---|---|---|---|
| `ibon` (writings/IBON, 6 PDFs) | 15 | 10 | 0.9333 → 0.9333 | 0.90 → 0.90 | yes, all 15 questions (5 off-topic included) |
| `fixtures` (test corpus, 5 files) | 12 | 15 | 0.9667 → 0.9667 | 0.9333 → 0.9333 | yes, all 20 questions |

The full hybrid index gives the same MRR on both (IBON 0.9333, fixtures
0.9667). **Verdict: kept.** M5-1 changes speed, not ranking, so equal MRR
is the expected result. These are the only collections available; there is
no real multi-document collection yet. M5-1 involves no reranker model, so
nothing was downloaded.

## M5-2 eval — 2026-11-12 (P2-9: idf-weighted gate coverage, closest material, search-only mode)

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

Keep-if rule: gate false-accepts down, false-rejects not up. **Gate part
reverted (FR went up). Closest material and search-only kept.**

**Labelled sets.** The M2-C pairs, re-measured on the current code: (a)
fixture eval on the clean collection, (b) the fixture questions asked of the
5 single-topic collections, (c) IBON's 10 on-topic + 5 off-topic questions,
(d) the 15 on-topic fixture questions asked of IBON. Because FA was already 0
on these for the hybrid gate, two harder off-topic sets that share words with
the corpus were written BEFORE the change and not edited afterwards: (e) 15
questions asked of the clean fixture collection ("How do I bake a molten lava
cake?", "Who is the current queen of Denmark?", …) and (f) 15 asked of IBON
("What is the minimum wage in Germany?", "Which jobs are best for
introverts?", …). Both gates were measured: hybrid (`gate_ok`, what the app
runs with vectors) and BM25-only (`bm25_gate_ok`, what a collection without
vectors runs). Scripts and sets: scratchpad `m5/gate_measure.py`,
`hard_fix.jsonl`, `hard_ibon.jsonl` (outside the repo).

**Tried: idf-weighted coverage** — each query term weighted by its BM25 idf,
and a term the corpus never contains weighted by the idf of df = 0. Same
MIN_COVERAGE (0.5), no threshold tuned.

| Gate, sets | FA before → after | FR before → after |
|---|---|---|
| hybrid, a–d (110 neg / 40 pos) | 0 → 0 | 0 → **2** |
| hybrid, a–f (140 neg / 40 pos) | 1 → **0** | 0 → **2** |
| BM25-only, a–d | 1 → **0** | 2 → **9** |
| BM25-only, a–f | 5 → **1** | 2 → **9** |

Fixture yardstick with the change: FR 0 → 1 ("What happens when a colony
loses its queen?"). FA falls as intended (the lava-cake hybrid false accept,
the glaciers "pyroclastic flows" BM25 false accept), but question words the
corpus lacks ("happens", "loses", "extracted") get the largest weight, so
on-topic questions whose content words are covered now fail coverage. That
is the same mechanism that rejects "capital of Mongolia", so it can't be
fixed by picking a different weight without tuning to these sets. Rule not
met → `rag.py` restored from the snapshot. The code is kept in the
scratchpad (`m5/rag_idfgate_tried.py`) for later.

**Kept (no gate change):**
- *Closest material*: a gated-out question returns `closest`, the 3
  nearest passages, one per section (`rag.closest_material`). It's in the
  `/ask` `meta` event, and the UI lists it under the refusal ("Closest
  material in this collection:"), each entry opening the reader. No model
  call. Not saved in the conversation file (`conversations.py` is not an
  M5-2 file), so a reopened gated turn shows only the refusal.
- *Search-only*: `POST /c/<id>/search {question, k?}` →
  `{question, covered, results[]}` (`rag.search_passages`), using the same
  pool → rerank → per-source cap as `/ask`, with no follow-up rewrite and no
  model call. `covered` is the gate's verdict, shown as a "weak match" note
  and never used to filter. UI: a "Search" button beside Ask. Rendering is
  DOM-only (textContent), with no innerHTML of corpus text.
- Both run after or beside `gate_ok` and never change what it decides:
  gate decisions are identical before and after on all 180 labelled pairs
  (sets a–f, both gates), and the fixture eval JSON is byte-identical
  (HybridIndex, hit@1 0.933, hit@3 1.0, hit@10 1.0, MRR 0.967, FA 0, FR 0).

Tests: `tests/test_m5_2_search.py` (14). UI checked with `node --check` and
a DOM-stub run of `passageList` / `renderAnswer` / `searchOnly` (scratchpad
`m5/ui_m52.js`). There was no browser click-through.

## M5-3 eval — 2026-11-12 (P2-3: heading segmentation for PDF/HTML/TXT/DOCX)

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

Keep-if rule: fewer sources on segmentation rung `none`, hit@k not lower.
**Kept.**

- PDF: new ladder rung `fonts` between `match` and `none`. It reads font
  size and weight through the pypdf text visitor and groups runs into lines
  by y. A heading line is all bold, or ≥ 1.2 × the body size (the size that
  carries the most characters). It must also be ≤ 12 words, contain
  letters, and not end in `. , ; :`. A candidate found on more than one page
  is a running header/footer and is skipped. Consecutive heading lines merge
  into one title.
- TXT: setext (`===`/`---` underline, dropped), `#`/`##`, ALL-CAPS and
  numbered (`2.1 …`, `Chapter 4`) lines, only at a paragraph start.
- HTML: `<h1>`/`<h2>`; heading text is taken out of the body.
- DOCX: Heading 2 added to Title/Heading 1.
- Every format needs ≥ 2 headings, else it stays `none` with its old
  output. Heuristic headings (txt, pdf `fonts`) must also head a real
  section: ≥ 40 words (= MIN_CHUNK_WORDS), with a prose line (≥ 6 words)
  among the next 3; otherwise the line stays body text. Thresholds were set
  before measuring, not tuned.
- `SEGMENTER_VERSION = 2` is recorded per source in the manifest. Sources
  chunked before it re-chunk once on the next plain ingest; vectors follow,
  because they are keyed by the chunk-file hash.

**Rung per source** (fresh scratch homes; fixture = all 8 `CORPUS_FILES`):

| Corpus | rung `none` before → after | chunks before → after |
|---|---|---|
| fixture (8 sources) | 4 → 4 | 16 → 16 |
| IBON (6 PDFs) | **6 → 4** | **14 → 15** |
| total | **10 → 8** | |

The fixture's 4 `none` sources have no headings to find:
`lighthouses.html` has a single `<h1>`, and `volcanoes.txt`, `report.txt`
and `tiny.txt` are plain paragraphs. They are correctly unchanged, and the
fixture chunks are identical. On IBON, 2 PDFs moved to `fonts`:
- "PH is upper middle-income…": real subheadings found ("Upper
  middle-class", "Borrowing scorecard", "Hostile witness"), 4 chunks
  (unchanged count).
- "Upper middle-income, Lower-end reality": sections are the nav front
  matter (61 words), "Upper middle-income, Lower-" (the article body, 833
  words), and "SONNY AFRICA" (author bio + footer, 43 words). The title is
  cut off because pypdf joins the title's second line with the date
  ("end realityJuly 6, 2026"), so it no longer matches the heading line, and
  the byline counts as a heading. Usable, not clean.
- **14 → 15 chunks** because of the second PDF: the 61-word nav and the
  43-word bio/footer are now their own chunks instead of sharing windows
  with the body.

The other 4 IBON PDFs are single web articles. Their only large heading is
the title, which pypdf places after page 1's body in content order, so they
correctly stay `none`.

**Retrieval** (`eval.py --json --variants` on IBON; `tests/eval_fixture.py`):

| | hit@1 | hit@3 | hit@10 | recall@10 | MRR | FA | FR |
|---|---|---|---|---|---|---|---|
| IBON bm25 before → after | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.967 → 0.956 | 0.933 → 0.933 | 0 → 0 | 0 → 0 |
| IBON hybrid before → after | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.967 → 0.956 | 0.933 → 0.933 | 0 → 0 | 0 → 0 |
| fixture (HybridIndex) | 0.933 → 0.933 | 1.0 → 1.0 | 1.0 → 1.0 | — | 0.967 → 0.967 | 0 → 0 | 0 → 0 |

hit@k did not drop. recall@10 dipped slightly because one more chunk now
matches an expectation pattern; it is not part of the keep-if rule. Fixture
eval JSON is byte-identical.

Not done: replacing `clean_page`'s odd-page header heuristic with
frequency-based header/footer stripping. It would change the `meta` and
`outline` rung output of existing books, and there is no real book
collection to measure it on. The `fonts` rung only uses frequency to reject
repeated heading candidates; it does not strip them from the text.

Tests: `tests/test_m5_3_segmentation.py` (11). Covers generated TXT, HTML,
DOCX and a two-font PDF with a running header, list items that are not
headings, the re-chunk on the old segmenter version, and no lost lines.

## M5-4 eval — 2026-11-12 (P2-4: paragraph-aware chunking with explicit offsets)

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

Keep-if rule: `passage()` round trip still exact, hit@k not lower. **Kept.**

- `ingest.chunk_spans`: a window is never longer than CHUNK_WORDS. Its end
  moves back by at most 10% of CHUNK_WORDS (`BOUNDARY_TOLERANCE`) to the
  latest paragraph start, else the latest sentence end, else it is cut at
  CHUNK_WORDS as before. The next window starts CHUNK_OVERLAP_WORDS before
  that end, moved forward to the first sentence/paragraph start inside the
  overlap, so overlap is ≤ the setting. Sentence ends skip initials and a
  short list of abbreviations (`St.`, `Dr.`, `e.g.`, …).
- Windows never lie inside the previous one, so the "drop a short
  redundant tail window" rule is gone and no word is lost.
- Each chunk stores `start`/`end` word offsets, and the id stays
  `<prefix>-<section>-<start>`. `rag.passage()` stitches with the offsets
  (`_overlap`: overlap = prev.end − start; a gap shows `[…]`). Chunk files
  without offsets (pre-P2-4) still use the fixed-step arithmetic.
  `SEGMENTER_VERSION` is now 3, so existing sources re-chunk once.
- Tolerance: 20% was the first choice. Simulated on `volcanoes.txt` at
  50/10, it makes 4 chunks where `tests/test_reader.py` pins 3, so 10% was
  used. 10% also cuts mid-sentence less often than 20% shortens chunks.

**Round trip** (every section, cited first/middle/last chunk, whole
section stitched vs the section's words re-extracted from the source):

| Collection, CHUNK_WORDS/overlap | exact before | exact after | chunks before → after | windows ending at a sentence |
|---|---|---|---|---|
| fixture (8 sources), 450/80 | 48/48 | 48/48 | 16 → 16 | 16 → 16 |
| fixture, 50/10 | 48/48 | 48/48 | 34 → 35 | 16 → 20 |
| IBON, 450/80 | 33/33 | 33/33 | 15 → 15 | 8 → 12 |
| IBON, 120/20 | 33/33 | 33/33 | 46 → 45 | 11 → 19 |

**Retrieval** (`eval.py --json --variants`; fixture also via
`tests/eval_fixture.py`):

| | hit@1 | hit@3 | hit@10 | recall@10 | MRR | FA / FR |
|---|---|---|---|---|---|---|
| fixture yardstick (450/80, HybridIndex) | 0.933 → 0.933 | 1.0 → 1.0 | 1.0 → 1.0 | — | 0.967 → 0.967 | 0/0 → 0/0 |
| fixture 50/10, bm25 | 1.0 → 1.0 | 1.0 → 1.0 | 1.0 → 1.0 | 0.817 → 0.817 | 1.0 → 1.0 | 0/1 → 0/1 |
| fixture 50/10, hybrid | 1.0 → 1.0 | 1.0 → 1.0 | 1.0 → 1.0 | 0.978 → 0.978 | 1.0 → 1.0 | 0/0 → 0/0 |
| IBON 450/80, bm25 | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.956 → 0.956 | 0.933 → 0.933 | 0/0 → 0/0 |
| IBON 450/80, hybrid | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.956 → **0.989** | 0.933 → 0.933 | 0/0 → 0/0 |
| IBON 120/20, bm25 | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.819 → 0.819 | 0.95 → 0.95 | 0/0 → 0/0 |
| IBON 120/20, hybrid | 0.90 → 0.90 | 1.0 → 1.0 | 1.0 → 1.0 | 0.856 → **0.881** | 0.95 → 0.95 | 0/0 → 0/0 |

Fixture yardstick JSON is byte-identical: every fixture section fits in one
450-word window.

Open issue (not an M5-4 file): `summarize.py:article_text` still trims
overlap with the fixed-step test (`start diff == CHUNK_WORDS −
CHUNK_OVERLAP_WORDS`). On re-chunked sources that test rarely holds, so
each section summary's input repeats the overlap words. That is harmless to
correctness but wasteful. Fix: stitch with `start`/`end` (as
`rag._overlap` does). Reported to the orchestrator.

Tests: `tests/test_m5_4_chunking.py` (14). Covers span coverage and size
limits for 5 settings, paragraph-over-sentence preference, the overlap
start snap, abbreviations, stored offsets, exact round trip (whole section
and ±1 window), paragraph breaks, the gap marker, a legacy fixed-window
chunk file, and the re-chunk on the old segmenter version.

## M5-5 eval — 2026-11-12 (P2-6: Unicode tokenizer, per-collection stopwords)

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

Keep-if rule: English numbers unchanged, non-ASCII fixture passes. **Kept.**

- `rag.tokenize`: NFKC, then casefold, then tokens are runs of Unicode
  letters/digits (`\w` minus `_`), combining marks (so Indic vowel signs
  and virama stay inside the word) and apostrophes. On ASCII text this is
  exactly the old `[a-z0-9']+` over `lower()`. Stemming and the
  length > 1 filter are unchanged. Accents are not folded: `été` ≠ `ete`,
  and all-caps Greek without tonos ≠ the accented form. CJK runs without
  spaces become one token each (not addressed).
- Per-collection stopwords: optional `<collection>/stopwords.txt` (one word
  per line, `#` comments, `-word` removes a default), normalized like the
  text. `BM25Index` carries its list (`.stopwords`, `.tokenize`). Query
  tokens, the gate's coverage and `SummaryIndex` all use it. The token-cache
  tag fingerprints the collection's list, so editing the file rebuilds the
  cache. It is a file rather than a setting because the settings schema
  lives in `config.py`, which is outside M5-5's files, so the settings
  API/UI can't edit it yet.
- `TOKENIZER_VERSION` 1 → 2, so every token cache rebuilds once.

**English numbers** (before → after):
- Fixture yardstick: byte-identical (HybridIndex, hit@1 0.933, hit@3 1.0,
  hit@10 1.0, MRR 0.967, FA 0, FR 0). Fixture chunks: 0 of 16 token lists
  changed.
- IBON (`eval.py --variants`): unchanged. bm25 and hybrid both hit@1 0.90,
  hit@3 1.0, hit@10 1.0, MRR 0.933, FA 0, FR 0; recall@10 0.956 (bm25) /
  0.989 (hybrid). IBON's PDFs use ﬁ/ﬂ ligatures, so 13 of 15 chunks' token
  lists did change ("ﬁnancial" was indexed as "nancial", now "financial";
  "classi"+"cation" → "classification"). On these 10 questions that changes
  no ranking.
- M5-2 gate pairs (180 labelled pairs, sets a–f, hybrid and BM25-only):
  every decision identical (hybrid FA 1/140, FR 0/40; BM25-only FA 5/140,
  FR 2/40).

**Non-ASCII fixture** (`tests/test_m5_5_tokenizer.py`, generated in the test;
not in `CLEAN_FILES`): French (accents), Greek (non-Latin) and German (ß,
ﬁ ligature) Markdown next to the 5 English fixture files, with 5 questions
(top-1 section + gate) and 2 off-topic ones (BM25 gate).
- Before (old `rag.py` in a scratch copy): 6 failed / 7 passed. Both Greek
  questions had no tokens at all.
- After: 13 passed, 1 xfailed.
- The strict xfail is a known limit that is not the tokenizer: the English
  bge-small encoder gives any Greek question cosine 0.73–0.77 to the Greek
  chunks, which is above `DENSE_SOLO_SIM` (0.72). The hybrid gate therefore
  accepts off-topic Greek ("Τι είναι η κβαντική φυσική;"), while the BM25
  gate rejects it. Fix: a multilingual `EMBED_MODEL` (bge-m3 /
  multilingual-e5-small, P2-6's second half). That needs a model download,
  so it was not done.

Tests: `tests/test_m5_5_tokenizer.py` (13 + 1 strict xfail). Also covers
ASCII equality with a verbatim copy of the old tokenizer, the stopwords
file (add/remove, cache tag, BM25 gate) and the version bump.

## M5-6 eval — 2026-11-12 (P2-7: ONNX reranker)

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

**Not run:** needs a reranker model download, which needs the user's
approval; keep-if (MRR up on a real collection) also needs a real
collection.

- `models/` holds only `encoder-onnx/` (bge-small-en-v1.5) and the Qwen
  GGUF. There is no cross-encoder ONNX model.
- A Hugging Face cache copy of `cross-encoder/ms-marco-MiniLM-L-6-v2`
  exists outside the repo (`/Volumes/APFS/LLMs/huggingface/hub/`), but it
  has `model.safetensors` only, no ONNX. Exporting it needs torch /
  sentence-transformers, which are not installed and may not be added.
  Using or exporting it is the user's call.
- No code changed for M5-6. `_maybe_rerank` still needs
  sentence-transformers, so reranking stays off in the bundle.

## M5-1 re-check (M5 agent) — 2026-11-12

Label: fixture + IBON only; no real multi-document collection yet
(EVAL_COLLECTION unset).

I reviewed P2-5 on the current code (after M5-2…M5-5): postings, memo,
`search_idx`, `doc_tokens_of`, the token cache, `bm25_gate_ok(hits=)` and
`gate_ok(bm25_hits=)`. The core holds:
- Memo keys are the index's own tokens.
- The cache is keyed by chunk text only, so M5-4's `start`/`end` fields
  don't touch it.
- The cache tag carries tokenizer v2 and the stopword fingerprint.
- `closest_material` / `search_passages` reuse the memoized scoring.

**Fixes (both from M5-5, with tests in `tests/test_m5_1_bm25.py`):**
1. Dense-only retrieval: the gate's fallback `BM25Index(dn.chunks)` ignored
   the collection's `stopwords.txt`. It now uses `collection_stopwords(col)`.
   Also, `collection_stopwords(None)` no longer reads a `stopwords.txt`
   from the working directory.
2. Speed: tokenizer v2's Unicode regex is about 3× slower than
   `[a-z0-9']+`, which pushed the uncached 50k build from 3.25 s to 4.89 s.
   ASCII text now takes an exact fast path (NFKC is the identity on ASCII,
   casefold equals lower, and the mark ranges are ≥ U+0300). A test checks
   it against the general path on 3,000 random ASCII strings plus the eval
   questions.

**Keep-if, re-measured on the current code:**
- Identical rankings on the fixtures: the reference-scorer tests pass (fixture
  questions at k = 1/3/10/all, and the 650-chunk random corpus with ties).
  `tests/eval_fixture.py` is byte-identical to M5-5's after-JSON (hit@1
  0.933, hit@10 1.0, MRR 0.967, FA 0, FR 0).
- Synthetic 50k chunks (`bench_bm25.py`; old = pre-M5 `rag.py` with
  tokenizer v1). Rank mismatches top-10: **0 / 200**. The corpus is ASCII,
  so v1 and v2 tokens are identical.

| | old | new (M5 final) |
|---|---|---|
| search k=10, median / p95 | 28.2 / 35.5 ms | **0.26 / 1.69 ms** |
| gate + search, median / p95 | 56.8 / 72.6 ms | **0.46 / 2.29 ms** |
| build, no cache | 2.75 s | 3.15 s (4.89 s before fix 2) |
| load, warm token cache | — | 2.70 s |
| load, cold (tokenize + write cache) | — | 3.88 s |

**Verdict: kept.** Rankings are identical, and search and the ask path are
about 110–120× faster. Load time with a warm cache is about the old
build's time, not faster (as the first M5-1 run found). These are single
runs on this machine.

## Citation prompt — 2026-11-13 (M3 numbered citations with real small models)

M3 numbered the excerpts `[1]…[n]` and asked for `[n]` markers in the
system prompt, but it had only been tested against a fake LLM. With real
1.5B models, most answers were uncited, and some were nothing but "[1]".
Two changes in `rag.py`:
- the excerpt header is `[1] Title — work, p. 3` instead of `[1] [Title —
  work, p. 3]`; the nested brackets were being copied into answers;
- numbered (scholar/persona) prompts end with `CITE_REMINDER`: answer in
  your own words, put the excerpt number after each sentence, a neutral
  worked example ("The bridge opened in 1932 [2]."), and "never reply with
  only a number". A first version used a bee example, and the bundled model
  copied it word for word into a bee answer, so the example must not come
  from any corpus.

Quotes mode is unchanged. Measured on the 15 on-topic fixture eval
questions, scholar mode, collection of the 5 clean fixture files:

| Backend | Answers citing correctly | Bare-marker answers | Header echoed |
|---|---|---|---|
| Ollama `qwen2.5:1.5b`, before → after | 3 → 14 | 1 → 0 | 0 → 0 |
| Embedded GGUF (desktop default), before → after | 10 → 15 | 4 → 0 | 1 → 0 |

Retrieval is unaffected: fixture eval unchanged (hit@10 1.0, MRR 0.967).
Not yet measured on larger models or a real collection; no fallback for
bare-marker answers was added, since none appeared after the change.

## JMS eval — 2026-11-19 (first real multi-document collection)

`jms` is the real collection the gate thresholds (M2) and the M5 items were
waiting for: 12 books (Jose Ma. Sison's writings), 4,257 chunks, linked from
`/Volumes/APFS/ClaudeAutomate/AskJ/writings`. Its eval set is
`collections/jms/data/eval.jsonl`: 25 on-topic questions, each with a regex
for the section or text that answers it, plus 10 off-topic questions. Two
patterns ("state of the Philippines", "basic problems") match 190–340
chunks because the corpus repeats those phrases; the rest match 4–175.
Measured on a scratch copy (`eval.py -c jms --json --variants`), current
thresholds:

| Variant | hit@1 | hit@3 | hit@10 | recall@10 | MRR | Gate FA / FR |
|---|---|---|---|---|---|---|
| BM25 only | 0.80 | 0.80 | 0.92 | 0.28 | 0.819 | 5 / 1 |
| **Hybrid (default)** | 0.80 | 0.92 | 0.96 | 0.24 | **0.858** | **1 / 1** |
| Hierarchical, +rerank | unavailable (no summaries; no reranker) | | | | | |

Gate signals (top-1 cosine): on-topic 0.696–0.884, off-topic 0.458–0.598,
so on JMS the dense score alone separates them with a wide margin.
- False reject: "What is the state of the Philippines?" Its words are
  everywhere in the corpus, so the BM25 top score is 3.18 against the
  scaled bar of 4.91, and its cosine is 0.716, just under `DENSE_SOLO_SIM`
  0.72 (the curly-apostrophe "what’s …" form scores 0.726 and passes).
- False accept: "How does photosynthesis work in plants?" BM25 passes and
  cosine is 0.598 ≥ `MIN_DENSE_SIM` 0.57.
- BM25-only gating is weak at this size: 5 of 10 off-topic questions pass.

Thresholds are **not** changed yet: lowering `DENSE_SOLO_SIM` to about 0.70
and raising `MIN_DENSE_SIM` to about 0.60 would fix both errors here, but
the second has a 0.002 margin, and the M2 labelled sets they must also be
checked against were lost with the scratch folder. Re-measure with a larger
JMS eval set before changing them. Run future M5 items with
`EVAL_COLLECTION=jms`.

## Follow-up history — 2026-11-19 (answers copied from earlier turns, JMS)

Bug from the JMS collection: with the previous Q&A in the prompt (M3,
P2-1), the bundled 1.5B model answered a new question with an earlier
answer word for word. "Does election bring real social change?" got the
previous "role of youth" answer (99–100 % 8-word-shingle overlap); a
follow-up copied an earlier "not covered" refusal the same way. History
was included on every turn of a conversation, not only on follow-ups.

Measured by replaying the user's 2 saved JMS conversations (10 questions)
with the embedded model (copied = ≥ 50 % of the answer's 8-word shingles
appear in an earlier answer):

| Variant | Copied answers | Cited |
|---|---|---|
| before (Q+A history on every turn) | 2 | 6 / 9 |
| V1: history only for follow-ups | 1 (copied a refusal) | 7 / 9 |
| **V2: follow-ups only, earlier questions only** | **0** | 7 / 9 |

V2 is kept: earlier answers never go into the prompt, and earlier
questions go in only when the question was rewritten as a follow-up.

The same replay exposed a second bug, from the "Citation prompt" fix
(2026-11-13): asked a question the excerpts barely covered, the model
answered with the reminder's example sentence ("The bridge opened in
1932 [2]."). Reminder wording, same replay:

| Reminder example | Cited | Example copied |
|---|---|---|
| "The bridge opened in 1932 [2]." | 7 / 9 | 2 |
| none ("a sentence taken from excerpt 2 ends with [2]") | 2 / 9 | 0 |
| **`"<a fact from excerpt 2> [2]"`** | **7 / 9** | **0** |
| `Format: "Your sentence here [2]."` | 5 / 9 | 1 |

With the kept wording, the 15 fixture questions give 14/15 cited on both
Ollama `qwen2.5:1.5b` and the embedded GGUF (was 14 and 15), with no bare
markers, header echoes or copied examples. Still open: the two election
answers stay off-topic. That is the retrieval miss already in the JMS eval
("Can elections bring real social change?"), not a copy.

## Gate thresholds on JMS — 2026-11-19 (MIN_DENSE_SIM 0.57 → 0.60, DENSE_SOLO_SIM 0.72 → 0.70)

The JMS eval set was extended to 96 questions: 63 on-topic, each tied to a
section, and 33 off-topic. 8 of the off-topic ones are hard negatives that
share words with the corpus in another sense ("political party in Canada",
"armies of ants", "state flower", "labor contractions", "youth soccer
league"). Thresholds were swept over 234 labelled questions in three
collections (scratch copies; top-1 cosine and the BM25 signal per
question):

| Set | On | Off | On-topic cos min | Off-topic cos max |
|---|---|---|---|---|
| JMS + fixture questions as negatives | 63 | 48 | 0.644 | 0.661 |
| IBON (new 15-question set) + fixture negatives | 10 | 20 | 0.657 | 0.525 |
| Fixtures + JMS/IBON questions as negatives | 15 | 78 | 0.614 | 0.537 |

IBON and JMS questions were not used as negatives for each other, since
both cover the Philippine economy.

| Rule (FA / FR) | JMS | IBON | Fixtures | Total errors |
|---|---|---|---|---|
| before: MIN 0.57, SOLO 0.72 | 11 / 1 | 0 / 0 | 0 / 0 | 12 |
| **after: MIN 0.60, SOLO 0.70** | **3 / 0** | 0 / 0 | 0 / 0 | **3** |

Any MIN 0.60–0.61 with SOLO 0.64–0.71 gives the same 3 errors. 0.60 / 0.70
leaves the most margin: the lowest on-topic fixture question (0.614) is
0.014 above MIN, "What is the state of the Philippines?" (0.716) is 0.016
above SOLO, and the highest off-topic score (0.661) is 0.039 below SOLO.
The 3 remaining false accepts are hard JMS negatives that pass BM25 with
cos 0.650–0.661 (ants, political party in Canada, and "ice cores … past
climate" asked of JMS); separating those needs a better BM25 signal on
large corpora, not a cosine threshold.

`eval.py -c jms` on the real collection (96 questions), hybrid: hit@1
0.89, hit@3 0.94, hit@10 0.98, recall@10 0.28, MRR 0.92; gate FA 9 → 2,
FR 1 → 0 ("before" is from the per-question gate dump on the same 96
questions). The fixture yardstick is unchanged (hit@10 1.0, MRR 0.967,
0 FA / 0 FR; `verify.sh M4` passes). Still the one retrieval miss: "Can
elections bring real social change?".

**Chunk length vs the encoder limit on JMS** (input to the open P0-2
`CHUNK_WORDS` decision): 3,345 of 4,257 chunks (79%) exceed bge-small's
512 tokens (median 543, p90 587, max 1,383), so 6.7% of all chunk tokens
are never embedded. `CHUNK_WORDS` ≈ 380 would fit most chunks; test it as
an M5-style item on JMS (re-ingest, keep if hit@k/MRR not lower).

## Chunk size on JMS — 2026-11-19 (kept CHUNK_WORDS 450)

JMS re-ingested on scratch copies with the overlap scaled to match (80 per
450 words). 96-question `eval.py -c jms`, hybrid:

| CHUNK_WORDS / overlap | Chunks | hit@1 | hit@3 | hit@10 | recall@10 | MRR | FA / FR |
|---|---|---|---|---|---|---|---|
| **450 / 80 (current)** | 4,257 | 0.89 | 0.94 | 0.98 | 0.28 | **0.92** | 2 / 0 |
| 380 / 67 | 4,993 | 0.83 | 0.94 | 0.97 | 0.27 | 0.88 | 2 / 0 |
| 320 / 56 | 5,896 | 0.81 | 0.92 | 0.98 | 0.26 | 0.88 | 3 / 0 |

Smaller chunks fit the 512-token encoder but rank worse: a 450-word chunk
carries more of its section's context, which outweighs the ~7% of tokens
past the limit that are not embedded. Verdict: keep 450. The P0-2
`CHUNK_WORDS` question is closed.

## M5-6 on JMS — 2026-11-19 (P2-7: ONNX reranker, kept)

Downloaded with the user's approval: `cross-encoder/ms-marco-MiniLM-L-6-v2`
as the repo's own pre-exported `onnx/model.onnx` (91 MB, no torch export),
fetched by the new `scripts/fetch_reranker.py` into `models/reranker-onnx/`.
`encoder.OnnxReranker` runs it on onnxruntime + tokenizers (pair encoding,
`only_second` truncation at 512); `encoder.get_reranker()` prefers it over
sentence-transformers' CrossEncoder, so reranking now works in the bundle.

`eval.py --variants`, +rerank over the hybrid base:

| Input to the cross-encoder | Pool | JMS MRR | JMS recall@10 | Fixtures MRR | IBON MRR |
|---|---|---|---|---|---|
| (no rerank: hybrid) | – | 0.92 | 0.28 | 0.97 | 1.00 |
| chunk text only | 60 | 0.86 | 0.25 | 1.00 | 1.00 |
| section heading + text | 60 | 0.95 | 0.37 | 1.00 | – |
| **section heading + text** | **20** | **0.96** | 0.36 | 1.00 | – |

Headings are stripped from chunk bodies, so text-only reranking lost the
section match that BM25 gets from `tokenize()`; with the heading the
reranker wins. Gate errors are unchanged (the gate does not use it).
Cost: about 45 ms per passage on CPU (60 passages 2.6 s, 20 about 0.9 s),
so `RERANK_POOL` = 20 (new setting) reorders only the top 20 candidates.
Verdict: keep (RERANK "auto" uses it when the model is present). The
bundle grows by 91 MB. Tests pin `RERANK` off (conftest) except the
reranker tests, so the suite still runs in about 45 s.

## JMS "elections" miss — 2026-11-19 (label, not retrieval)

"Can elections bring real social change?" was the one JMS miss in every
variant. Retrieval was right; the eval label was too narrow. The top
hybrid hit ("Consolidate the People's Gains in the Electoral Struggle")
says "it is not possible to make a fundamental change of the ruling system
through periodic elections", and rank 6 says "Elections under the auspices
of the US-Marcos clique cannot bring about any fundamental change". Neither
matched the pattern. The pattern now also accepts those passages and
"electoral struggle". Real JMS collection after the fix: hybrid hit@10
1.00, MRR 0.93; +rerank MRR 0.97.

## PDF running headers — 2026-11-19 (opt-in, off by default)

Survey of the 12 JMS PDFs: 1,681 of 4,450 pages start or end with a line
repeated across the document. `clean_page`'s odd-page rule removed some;
1,068 pages kept one in the chunk text ("Building Strength Through
Struggle", "ó ó ó" dividers, "Page 3/120").

`ingest.repeated_edge_lines` / `strip_edge_lines`: a line in the first or
last 2 lines of a page, normalized (lowercase, digits → #), that repeats at
the edge of at least 4 pages and 3% of the document is stripped from page
edges. Date lines ("12 May 1986") and digit-only lines other than a bare
page number are kept. JMS re-ingested with it, 96 questions:

| | Header phrases in chunk text | hybrid MRR | +rerank MRR | Gate FA |
|---|---|---|---|---|
| off (current) | 724 | 0.93 | 0.97 | 2 |
| on | 89 | 0.92 | 0.97 | 3 |

It does not help ranking: a running header usually repeats the section
title, and that repetition helps BM25 ("How did Marcos usurp absolute
power?" fell from rank 3 to 7). An off-topic question ("book club")
rose from cosine 0.577 to 0.604 and passed the gate. Verdict: kept as the
per-collection setting `PDF_STRIP_HEADERS` ("off" by default; "on"
re-chunks that collection), for collections whose quotes and reader
display suffer from header lines. Existing collections are not re-chunked.

## P4 items — 2026-11-19

- **GPU:** `LLM_GPU_LAYERS` (default -1 = all layers; 0 = CPU). Same
  1,729-token prompt + ~150-token answer on this Mac (Metal): CPU 8.3 s,
  GPU 3.7 s; model load 1.0 s → 0.2 s. If GPU loading fails, the model
  loads on the CPU.
- **App icon:** `scripts/make_icon.py` draws `assets/anyrag.png` and
  `assets/anyrag.icns` with no image library (stdlib PNG + sips/iconutil).
  `anyrag.spec` passes it to both the executable and the `.app` bundle
  (before, only to the executable, so the bundle showed PyInstaller's icon).
- **First-run model download:** `modeldl.py`. The bundle no longer contains
  the GGUF unless built with `ANYRAG_BUNDLE_MODEL=1`. `config.LLM_MODEL_PATH`
  is the bundled file when present, else `<data>/models/<gguf>`. The UI
  shows a banner and a Settings button: progress, cancel, resume (HTTP
  Range on `.part`), SHA-256 check against the Hugging Face hash. The macOS
  bundle went from ~1.5 GB to 362 MB (with the 91 MB reranker). Tested
  against Hugging Face: 146 MB at 24 MB/s, cancel kept the part, resume
  continued from it.
- **Log:** `applog.py`, `<data>/logs/anyrag.log` (1 MB × 3). Settings → Log:
  "Open log folder", "Show recent log".
- **CI:** `.github/workflows/ci.yml`. Tests on Linux/macOS/Windows; then
  PyInstaller builds on all three and `AnyRAG --smoke <fixtures>`
  (`smoke.py`): ingest, search, gate, encoder, reranker, llama.cpp
  library, HTTP with the token cookie. Not run on GitHub yet (nothing
  pushed). Locally: the frozen macOS app passes `--smoke`. A clean venv
  with only `requirements.txt` + dev deps fails 8 tests that assume the
  dense encoder, so CI installs numpy/onnxruntime/tokenizers and fetches
  the encoder; with those, 320 passed.
- **BM25-only gap found by that run:** "And what about those?" passes the
  BM25-only gate on the fixtures because "those" is not a stopword. Hybrid
  installs reject it. Adding demonstratives to `STOPWORDS` changes BM25
  scoring everywhere, so it needs its own eval; not changed.
- **HTTP drive** of the source server with the embedded model (25 checks,
  all pass): token cookie, settings (new keys, validation), linked
  collection create + streamed ingest, search, ask with numbered citations
  (2.0 s on GPU), a follow-up rewritten without copying (1% 8-gram
  overlap), off-topic gated, conversation saved, `/logs`, and the download
  flow above. It also exposed a log bug: the crash hook replaced Python's
  default, so dev crashes stopped reaching the terminal; the hooks now
  chain.

## Function-word stopwords and BM25-only tests — 2026-11-19

Follow-up to "P4 items". Added to `STOPWORDS`: these those there here
then than so if also been being am me my him she them into such very just
each. Kept out on purpose: "us" (casefolded "US") and "may" ("May 1"),
both JMS search terms. The token cache rebuilds by itself, because its tag
hashes the stopword list.

| Set | Variant | MRR before → after | Gate FA / FR before → after |
|---|---|---|---|
| JMS (96) | bm25 | 0.89 → 0.89 | — |
| JMS | hybrid | 0.93 → 0.93 | 2 / 0 → 2 / 0 |
| JMS | +rerank | 0.97 → 0.98 (hit@1 0.95 → 0.97) | 2 / 0 → 2 / 0 |
| Fixtures, IBON | all | unchanged | unchanged |
| 234 labelled incl. cross negatives | hybrid gate | — | JMS 3/0, IBON 0/0, fixtures 0/0 (unchanged) |
| same | BM25-only gate | — | JMS 28/1 → 29/1, others unchanged |

The one BM25-only change: "How does snow turn into glacial ice?" asked of
JMS now passes ("into" dropped, so coverage rises). BM25-only gating on
large corpora was already weak (28 of 48 negatives), and hybrid is
unaffected. "And what about those?" now tokenizes to nothing, so a vague
follow-up is gated out in every mode, as the follow-up rewrite expects.

The 7 tests that failed on a BM25-only install (requirements.txt only)
assumed dense retrieval. Now each holds in both modes:
- `tests/helpers.py`: `has_dense()` and `gate_expected()`, with the known
  BM25-only false reject (ice cores, `tests/test_gate.py`).
- `test_m1d`: skips the vector part without numpy.
- `test_m3a`/`test_m4a`: BM25-only ranks only chunks that contain a query
  term, so the expected excerpt count is min(TOP_K, matching chunks).
- `test_m5_2`: the off-topic question now shares one word ("volcanoes")
  with the fixtures, so closest material exists in both modes.
Results: dense install 322 passed; BM25-only venv 301 passed, 22 skipped
(they need the encoder). CI has a new `bm25-only` job.

## Reranker veto — 2026-11-19 (the two hard JMS negatives)

The signal gate (BM25 + cosine) could not reject "How do armies of ants
organize their colonies?" (cos 0.650) and "How do I join a political party
in Canada?" (0.655): they share words with JMS, and real questions start at
0.644. The cross-encoder reads question and passage together. Its best
score over the candidate pool, on the 234 labelled questions:

| Set | Real questions, lowest | Off-topic, highest |
|---|---|---|
| JMS (+ fixture questions as negatives) | 0.21 | −1.62 (Canada) |
| IBON | −0.23 | −10.79 |
| Fixtures (+ JMS/IBON negatives) | −1.09 | −8.75 |

`rag.rerank_veto_ok`: when the signal gate accepts, the question is still
"not covered" if the best reranker score is below `RERANK_MIN_SCORE`
(−1.35, midway between −1.62 and −1.09). `RERANK_GATE` "auto"/"off". It
never accepts what the signals reject. ms-marco is English-only: it gives
every Greek question about +9, an English question over a French passage
about −10, and the keyword query "STRASSENBAHN ZÜRICH" −1.9. So the veto,
and now also the rerank reordering, apply only to an English sentence (≥ 3
words, ≥ 10% English function words) whose top passage is English (≥ 12%;
English chunks score ≥ 0.14, with 2 of 4,257 JMS chunks below; French and
Greek 0, German 0.10). Scores are memoized, so the gate and the answer
share one reranker run (about 0.7 s on JMS).

| Gate FA / FR | JMS (111) | IBON (30) | Fixtures (93) | Mixed-language (37) |
|---|---|---|---|---|
| before (signals) | 3 / 0 | 0 / 0 | 0 / 0 | 3 / 4 |
| **+ veto** | **0 / 0** | 0 / 0 | 0 / 0 | 3 / 4 |

The user's real JMS collection (96 questions): FA 2 → 0, FR 0; hybrid MRR
0.93, +rerank 0.98.

## Multilingual model — 2026-11-19 (intfloat/multilingual-e5-small, opt-in)

Downloaded with the user's approval (470 MB ONNX, pre-exported in the repo):
`scripts/export_onnx.py --model intfloat/multilingual-e5-small`. New
per-model `config.ENCODER_PROFILES`: ONNX folder, query/passage prefixes
(e5: "query: " / "passage: ", applied at query time and at embed time),
and the model's own gate cosines, which apply unless the user sets them.
The cached encoder reloads when `EMBED_MODEL` changes. Test set: a
37-question mixed-language collection (the French/Greek/German test
documents plus the English fixtures; same-language, cross-language and
off-topic questions in 4 languages), plus JMS, IBON and the fixtures
re-embedded with e5.

| MRR | bge hybrid | e5 hybrid | bge +rerank | e5 +rerank |
|---|---|---|---|---|
| Mixed-language | 0.83 | 0.90 | 0.86 | **0.92** |
| JMS (English) | **0.93** | 0.87 | **0.98** | 0.97 |
| Fixtures / IBON | 0.97 / 1.00 | 0.97 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

e5's cosines are compressed: unrelated text sits at 0.72–0.85, so it gates
English poorly without the veto (JMS ≥ 11 false accepts at any
threshold). With the veto, its swept gate (MIN 0.75, SOLO 0.77) gives
JMS 0/0, IBON 0/0, fixtures 0/0 and mixed-language 1/1. bge on the mixed
set: 3/4. Off-topic Greek drops to cos 0.754–0.756 against 0.82–0.89 for
real Greek questions. Remaining: "Quelle est la recette du gâteau au
chocolat ?" (cos 0.794, French, so no veto) and "When was the Parthenon
built on the Acropolis?" (hybrid ranks English passages above the Greek
one, and the English reranker cannot see the match).

Verdict: bge stays the default. Every current collection is English, where
bge ranks better. e5 is the setting for collections in other languages or
mixed languages (`EMBED_MODEL`, then re-ingest). Its ONNX folder is not in
the desktop bundle; in the packaged app, switching needs the folder in
`models/` (a first-run download like `modeldl.py` would be the follow-up).

## Answer prose — 2026-11-20 (reported in the desktop click-through; fixed)

Reported by the user: scholar answers read "Excerpt [1] says … Excerpt [2]
elaborates …", a summary per excerpt, and always come back as one
paragraph. Seen in the saved JMS conversation 20261120-052155: 2 of 9
covered answers name "Excerpt [n]" in the prose, one walks excerpts 1–6 in
turn, and 9 of 9 are a single paragraph. Likely cause: `CITE_REMINDER`
(the last thing the model reads) asks for "a few sentences" and its
example is `"<a fact from excerpt 2> [2]"`. Nothing asks for paragraphs.
Wanted: coherent paragraphs, one idea each, with only the bracket marker
where a claim comes from.

Prompt attempts (10 JMS questions, embedded Qwen2.5-1.5B, user's real
collection). Sentences are counted with the marker after the full stop
attached to them.

| Prompt | Multi-paragraph | Answers with no citation | Notes |
|---|---|---|---|
| current (baseline) | 0/10 | 4/10 | 26 of 111 sentences uncited |
| + paragraph and "no Excerpt" rules, longer reminder | 0/10 | 9/10 | rules cost citations |
| B: layout example in the reminder | 4/10 | 0/10 | loops ("[2]. [1]. [2]. …"), 405 sentences |
| C: same, reminder only | 1/10 | 4/10 | |
| D: one "new paragraph" sentence added | 0/10 | 7/10 | 2/10 name "excerpt" |

The 1.5B model can't hold layout rules: every addition cost citations
or made it loop. So the prompt is unchanged and the shape is fixed after
generation instead: `rag.compose_answer(text, excerpts)`, applied to
scholar and persona answers (never quotes) in `serve.py` (a `final`
NDJSON event replaces the streamed text; the saved turn holds the
composed form), in `rag.ask` and in `ask.py` transcripts. It never adds
or rewords a claim. What it does:

- **Narration → marker:** "Excerpt [4] mentions that X." → "X [4]."; "…,
  as discussed in Excerpt [1], …" and "According to excerpt 3, …" are
  dropped and become the sentence's marker. "Excerpt [3] discusses Y" →
  "The source discusses Y [3]." "in the excerpts" with no number is
  dropped.
- **Markers** go at the end of their sentence, before the full stop, with
  no repeats ("X. [2]. [1]" → "X [2][1].").
- **Repeats:** exact repeated sentences are dropped (the looping answers).
- **Paragraphs:** the model's breaks are kept after a paragraph of 2+
  sentences. One sentence per line is merged. A new paragraph starts at a
  topic change: under 15% of a sentence's content words are shared with
  the previous two sentences, or under 40% when the cited excerpts also
  change. At most 5 sentences per paragraph. No break before a sentence
  that opens with "It/They/This/However/…". Lists, headings, quotes,
  tables and code are left as written.
- **Lifted sentences:** calibration showed the model's own numbers are
  mostly wrong. It numbers sentences [1], [2], [3] in turn while copying
  them from one excerpt: on 28 answers, only 13 of 151 sentences had a
  marker whose excerpt contained at least half the sentence's content
  words. A sentence counts as lifted when at least 80% (`LIFT_COVER`) of
  its content words are in one excerpt (4 or more words). If it has no
  marker, it is cited to that excerpt. If its marker's excerpt covers
  under 50% and the lifted-from one covers 0.3 more, the marker is
  replaced. A title citation "[Section, p. N]" is left alone. Paraphrases
  keep the model's choice.

| 28 JMS answers | Cited, and the excerpt supports it | Uncited sentences | Answers with no citation |
|---|---|---|---|
| model output | 9% (13/151) | 62% | 11/28 |
| + compose_answer, no excerpts | 17% (19/114) | 69% | 11/28 |
| **+ lifted-sentence citations** | **77% (88/114)** | **18%** | **1/28** |

End to end on the 10 questions, before → after: 0 → 5 answers in more than
one paragraph (the other 5 are 1–3 sentences), 6 → 0 with markers after
the full stop, 2/9 → 0 naming "Excerpt" (the user's saved session).
Tests: `tests/test_p7_answer_prose.py`.

