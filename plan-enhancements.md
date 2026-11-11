# AnyRAG — Enhancement Plan

Review date: 2026-11-11 (branch `dev-features`, HEAD `d29a993`). Companion to
`plan.md` (the original phased build); this file lists what to do next.

## Current state

**Working (checked in this review):** the one dev collection (`ibon`, 6 linked
PDFs, 14 chunks) loads as `HierarchicalIndex` over BM25 with summaries; the
relevance gate passes an on-topic question and rejects two off-topic ones.
Desktop deps are installed in `.venv`, `models/` holds the GGUF and the ONNX
encoder, and `dist/AnyRAG.app` exists.

**Not exercised yet (per `plan.md` and what is on disk):**

- Dense/hybrid retrieval has never run: `data/vectors/` is empty and the
  manifest says `skipped (sentence-transformers not installed)`, even though
  the ONNX encoder and numpy are present (see P0-1).
- No `eval.jsonl` exists for any collection and `plan.md` records no eval
  numbers, so there is no retrieval baseline.
- No tests, no CI (`.github/` is absent, although Phase 1 and Phase 5 planned
  both).
- `plan.md` status is dated 2026-07-16 and still lists the desktop build as
  untested; `CLAUDE.md` is untracked.

**Planned in `plan.md` / `design.md` but not in the code:**

| Planned | Actual |
|---|---|
| Settings UI (Phase 6) | `GET /settings` only; no write route, no UI. Error text tells users to "Check Settings". |
| Native folder picker (Phase 5) | Wizard takes a typed path; `app.py` exposes no `js_api`. |
| Source manager: remove, re-ingest one source (Phase 4) | Title/years edit only. No route removes a source. |
| Edit a collection after creation | No route; name, description, author and persona are fixed at create time. |
| Heading heuristics for txt/html (Phase 2) | `.txt` and `.html` are always one section; a PDF with no TOC/outline falls to `none`. All 6 IBON PDFs are `none`. |
| Flag scanned PDFs (design non-goals) | A text-less PDF silently yields 0 chunks. |
| Per-collection settings | Only `RETRIEVAL`, `CHUNK_WORDS`, `CHUNK_OVERLAP_WORDS` go through `col.setting()`; everything else reads `config.*`. |
| Staleness "never mtime" | Chunk stage uses mtime (`ingest.py:479`). |

---

## P0 — Correctness bugs (fix before adding features)

### P0-1. Desktop build can't embed linked collections
`ingest.embed_source()` imports `sentence_transformers` only (`ingest.py:601`),
which `anyrag.spec` excludes. The only ONNX embed path is the single-file
`serve.import_work()`. So in the shipped app every folder-linked collection
(the default UX) is BM25-only, and the bundled encoder is dead weight.

- Add one `get_encoder()` (ONNX if bundled, else sentence-transformers) and use
  it in `embed_source`, `import_work`, `DenseIndex` and `SummaryIndex`.
- Delete the duplicated embed code in `import_work`; call `embed_source`.

### P0-2. ONNX encoder likely doesn't match the sentence-transformers one
`OnnxEncoder` mean-pools and truncates at 256 tokens (`rag.py:160,178`); its
docstring still says MiniLM. `bge-small-en-v1.5` is a CLS-pooled model with a
512-token limit. The parity assert in `scripts/export_onnx.py` only runs when
sentence-transformers is installed, which it isn't here, so this has never
been checked. Corpus vectors from one encoder and queries from the other would
quietly degrade retrieval.

- Run the cross-check once with sentence-transformers installed; fix pooling
  and `max_length` to match.
- Store pooling and max length in the vector tag alongside `model`.
- Note that 450-word chunks (~600 tokens) exceed 512 tokens either way: the
  tail of every full chunk is not embedded. Either cut `CHUNK_WORDS` to ~350
  or accept it knowingly.

### P0-3. The dense gate probably passes everything
`gate_ok()` ORs the signals, and the dense branch passes at
`cos * 20 >= MIN_SCORE`, i.e. cosine ≥ 0.25 (`rag.py:226,438`). bge cosines for
unrelated text usually sit well above that. `plan.md` says the dense gate
"tightens" the BM25 one, but an OR can only loosen it. Untested because dense
has never run.

- Measure cosine for on-topic vs off-topic questions, then give dense its own
  threshold (`MIN_DENSE_SIM`) instead of reusing `MIN_SCORE`.
- Consider requiring both signals when both exist, or use the reranker score
  as the gate.

### P0-4. Removed or renamed sources stay in the index forever
`run_pipeline()` only iterates current source files. Nothing deletes
`chunks/paras/dates/quotes/summaries/vectors` or the manifest entry for a
source that has disappeared, and `load_sources()` globs every chunk file. A
file deleted from a linked folder keeps answering questions.

- At the start of ingest, prune derived artifacts whose source is gone; emit a
  `prune` progress event.

### P0-5. Same-stem sources overwrite each other
Derived files are keyed by `path.stem`, so `report.pdf` and `report.docx`
both write `chunks/report.jsonl`. Key by full filename (or stem + short hash)
and migrate on next ingest.

### P0-6. Context can overflow the model window
Top-k 10 × 450 words ≈ 6,000+ tokens of excerpts, plus system prompt, plus
1,024 output tokens, against `LLM_CTX = 8192`. The embedded backend raises on
overflow; Ollama silently truncates the prompt head (which is where the
system prompt is). There is no token budgeting.

- Budget `build_context()` to `LLM_CTX − LLM_MAX_TOKENS − prompt overhead`,
  dropping lowest-ranked hits; report the dropped count in `meta`.
- `LLM_MAX_TOKENS` is sent only to Ollama; send `max_tokens` to the `openai`
  and `embedded` backends too (`rag.py:516,533`).

### P0-7. Embedded LLM is not thread-safe under the threaded server
One shared `Llama` instance, `ThreadingHTTPServer`, no lock. Two simultaneous
questions (or a question during summarization) will corrupt or crash.
Wrap embedded generation in a module lock.

### P0-8. Un-ingested collections kill the request thread
`load_sources()` raises `SystemExit` when a collection has no chunks. In
`serve.py` that escapes `except Exception`, so `GET /c/<id>/stats` drops the
connection with no response; the UI only works because `fetch` throws.
Raise a real `NotIndexed` exception and return a 409 JSON error. Keep
`SystemExit` in the CLIs only.

### P0-9. Smaller defects
- Documents under 40 words produce zero chunks (`ingest.py:493`); keep the
  first chunk of a section regardless of length.
- Duplicate section titles in one PDF are merged into one section
  (`sections.setdefault(title, …)` in `gen_auto`).
- AskJ leftovers: `during martial law → (1972, 1981)` in `_ERA_PATTERNS`
  (`rag.py:652`); "not the author himself" in `conversations.to_markdown`;
  "land reform and national democracy" test string in `export_onnx.py`.
- Chunk staleness is mtime-based, contradicting the documented invariant;
  store the source sha256 in the manifest and compare that.
- `_RUNTIMES` is evicted in insertion order, not LRU, and `_RUNTIME_LOCK` is
  held while an index loads, blocking every other request.
- `collections.json`, `collection.json` and `manifest.json` are written
  non-atomically; write to a temp file and `os.replace`.

---

## P1 — Safety net (cheap, unlocks everything else)

### P1-1. Baseline eval, then record numbers
Write 20–30 `eval.jsonl` questions for a real multi-document collection
(IBON's 14 chunks are too small to discriminate). Record BM25 / hybrid /
hierarchical / +rerank in `plan.md`. P0-2, P0-3 and every P2 retrieval item
depend on this.

Extend `eval.py`:
- MRR and recall@k, not just hit@k.
- A `"expect": null` convention for off-topic questions, to score the gate
  (false-accept and false-reject rates).
- `--json` output so runs can be diffed.

### P1-2. Minimal test suite
`pytest` against a temp `ANYRAG_HOME` with tiny fixture docs (md, txt, html,
docx, a small PDF). Stdlib-only paths need no model:
- registry CRUD and collection isolation;
- chunk → reader round trip (`passage()` stitches back the original text);
- `verify_quotes` accepts verbatim, rejects altered;
- gate behaviour on a fixed corpus;
- prune-on-rescan (P0-4) and stem collisions (P0-5);
- HTTP smoke test via `serve.make_server(0)`.

### P1-3. Local-server hardening
The server trusts any request that reaches the port. A web page the user
visits can POST to `127.0.0.1:8000` (the JSON body is parsed regardless of
content type), e.g. create a collection linked to an arbitrary folder, and DNS
rebinding would let it read answers back.
- Reject requests whose `Host` is not `127.0.0.1`/`localhost:<port>` and whose
  `Origin` (when present) is not the app's own.
- Require `Content-Type: application/json` on JSON POSTs.
- In `app.py`, pass a per-launch token to the window and require it.
- Cap `_body()` size.

---

## P2 — Retrieval and answer quality

### P2-1. Multi-turn follow-ups
Every question is sent alone; `--chat` and the web conversation have no
memory, so "what about after 2020?" retrieves nothing useful. The
conversation schema already stores what is needed ("Option C").
- Rewrite the follow-up into a standalone query using the last 2–3 turns
  (one short LLM call, skipped when the question has no pronouns/ellipsis),
  retrieve on the rewrite, and include the recent turns in the prompt within
  the P0-6 budget.

### P2-2. Numbered, verifiable citations in scholar mode
Citations are free text the model writes (`[Section, p. N]`); nothing checks
them and the UI lists sources separately. Number the excerpts `[1]…[n]`, ask
for `[n]` markers, then map them mechanically: clickable inline markers that
open the reader, a warning on out-of-range markers, and unused sources
dimmed. Small models follow `[n]` far more reliably than long titles.

### P2-3. Segmentation for documents without a TOC
All six IBON PDFs landed on rung `none`, which makes "section" equal
"document" and hierarchical retrieval a no-op.
- PDF: heading detection from font size/weight (pypdf visitor) or
  ALL-CAPS/numbered short lines.
- HTML: segment on `h1`/`h2` (the parser already sees them).
- TXT: setext/ALL-CAPS/numbered heading heuristics.
- DOCX: include Heading 2.
- Strip repeated header/footer lines by frequency across pages rather than
  the odd-page heuristic in `clean_page`, which was tuned for one book.

### P2-4. Paragraph-aware chunking
Chunks are fixed 450-word windows that cut mid-sentence. Break at the nearest
paragraph/sentence boundary inside a tolerance (the `paras` data is already
computed). This changes chunk ids and the reader's fixed-`step` overlap
arithmetic, so store explicit `start`/`end` word offsets in each chunk and
have `passage()` use them. Measure with P1-1.

### P2-5. BM25 that scales
`BM25Index.search()` scans every chunk per query, the gate runs a second
scan, and the index is re-tokenized on every load. Fine at 14 chunks, slow at
100k.
- Build an inverted index (term → postings) and score only matching docs.
- Reuse the search hits inside `gate_ok` instead of searching again.
- Cache tokenized docs on disk keyed by chunk-file hash.

### P2-6. Tokenizer beyond ASCII English
`tokenize()` keeps `[a-z0-9']` only, so accented and non-Latin text is
dropped, and stopwords are English-only. Use `\w` with Unicode casefolding and
NFKC normalization; make the stopword list per-collection. For multilingual
corpora, offer `bge-m3`/`multilingual-e5-small` as `EMBED_MODEL`.

### P2-7. Reranker in the desktop build
`_maybe_rerank` needs sentence-transformers, so it is always off in the
bundle. Export `ms-marco-MiniLM-L-6-v2` to ONNX (plan Phase 6 already says
so) and run it through onnxruntime. Keep only if eval shows a gain.

### P2-8. Make all tuning per-collection
Route `TOP_K`, `MIN_SCORE`, `MIN_COVERAGE`, `MAX_PER_WORK`, `HIERARCHICAL`,
`RERANK`, `TEMPERATURE` and the quotes settings through `col.setting()`.
`MAX_PER_WORK = 6` in particular is wrong for a one-document collection and
for a collection of hundreds of short articles.

### P2-9. Gate refinements
- The known "capital of Mongolia" false accept: weight coverage by idf so a
  match on one common term does not count as half the question.
- When gated out, return the nearest section titles ("closest material: …")
  instead of a bare refusal.
- Add a "search only" mode that returns ranked passages with no LLM call.

---

## P3 — Product and UI

### P3-1. Settings (finish Phase 6)
`POST /settings` and `POST /c/<id>/settings` with validation, plus a settings
panel: backend, model, URL, temperature, top-k. Add a "Test connection"
button and, for Ollama, a model dropdown from `/api/tags`. Apply without
restart (today `config` is read once at import).

### P3-2. Collection and source management
- Edit name/description/author/persona after creation.
- Remove a source; re-ingest one source.
- Show per-source status and errors from the manifest (the `error` field is
  written but never displayed) and flag scanned/empty PDFs.
- Recurse into subfolders of a linked folder (currently top level only), with
  ignore patterns.
- Confirm before deleting a collection; show its disk usage.

### P3-3. Native folder picker and drag-and-drop
Expose `webview.create_file_dialog(FOLDER_DIALOG)` through a `js_api` for the
wizard; multi-file drag-and-drop import for managed collections.

### P3-4. Ingest UX
- A real progress bar: events carry `total` but no running `done` count, and
  the UI just appends log lines.
- Run ingest in a background job with `GET /c/<id>/ingest/status` so closing
  the tab does not orphan it and a second tab can attach.
- Cancel button.
- First-run model status: detect "Ollama not running / model not pulled /
  GGUF missing" up front instead of failing on the first question.

### P3-5. Chat UX
- Render Markdown in scholar/persona answers (currently a raw text node).
- Stop button (`AbortController`; the server already tolerates
  `BrokenPipeError`).
- Regenerate, copy answer, per-question mode switch.
- Search across conversation history; rename conversations.
- Reader: open the original file at the cited page (`file://…#page=N` for
  PDFs) as well as the stitched text.

### P3-6. More formats
EPUB (planned "later" in `design.md`), `.pptx`, `.xlsx`/`.csv` (row-wise),
`.rtf`, `.eml`. Optional OCR for scanned PDFs behind a flag
(`ocrmypdf`/Tesseract when installed).

### P3-7. Cross-collection search
A design non-goal for v1, but the index classes already take a chunk list:
an "All collections" pseudo-collection that fuses per-collection results with
RRF would be a small change once P2-8 lands.

---

## P4 — Packaging and housekeeping

- **CI** (`.github/workflows/`): pytest on push; build matrix
  macOS/Windows/Linux with the Phase 5 smoke test (launch, create collection
  from fixtures, `POST /ask`).
- **Bundle size:** make the GGUF a first-run download with a progress bar, so
  the installer drops from ~1.2 GB to ~150 MB and model choice becomes a
  setting rather than a rebuild.
- **macOS:** signing and notarization; app icon (`assets/anyrag.icns` is
  referenced by the spec but missing).
- **GPU:** `n_gpu_layers=-1` for the embedded backend on Metal builds.
- **Logging:** `log_message` is a no-op and failures print to a console the
  packaged app doesn't have. Write a rotating log in the user-data dir and add
  "Open log folder" to the UI.
- **Schema version** in `collection.json` and the manifest, so P0-5 and P2-4
  can migrate old collections instead of forcing `--force`.
- **CLI:** `anyrag create|list|delete|ingest|ask` subcommands replacing the
  `python -c "import library; …"` incantation.
- **Docs:** refresh the `plan.md` status block (desktop build exists; dense
  still unverified), fix the `collections.py` → `library.py` name in
  `design.md`/`plan.md`, remove the mtime contradiction, commit `CLAUDE.md`.

---

## Suggested order

1. **P1-1** baseline eval on a real collection, and **P1-2** fixtures + tests.
2. **P0-1, P0-2, P0-3** together: get dense retrieval actually running on the
   ONNX path, verify parity, re-tune the gate, record numbers.
3. **P0-4 … P0-9** index hygiene and server robustness.
4. **P1-3** server hardening.
5. **P2-1, P2-2** follow-ups and numbered citations (largest visible quality
   gain).
6. **P3-1, P3-2, P3-4** settings, source management, ingest UX.
7. **P2-3 … P2-9** retrieval improvements, each gated on eval numbers.
8. **P4** packaging, when a release is in sight.
