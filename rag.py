"""RAG core: retrieval over chunked sources + generation via a configurable
local LLM backend (Ollama or any OpenAI-compatible server).

Retrieval (config.RETRIEVAL):
  bm25    pure-Python BM25 (stdlib only) — always available
  dense   encoder.py vectors (data/vectors/; ONNX or sentence-transformers)
  hybrid  BM25 + dense fused with reciprocal-rank fusion (RRF)
  auto    hybrid when vectors + libs exist, else bm25

Generation supports streaming (llm_stream / ask_stream) and blocking (ask).
"""
import heapq
import json
import math
import re
import threading
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import config
import encoder
import errors

STOPWORDS = set("""a an and are as at be but by for from has have he her his i in is it its
of on or our that the their they this to was we were which with you your not no
what why how when where who whom whose do does did can could should would
say says said tell explain describe about""".split())


def _light_stem(w):
    """Cheap plural/possessive folding so 'rockets' matches 'rocket'.
    Deliberately conservative: no -ing/-ed stripping (too many false folds)."""
    if w.endswith("'s"):
        w = w[:-2]
    if (len(w) > 3 and w.endswith("s")
            and not w.endswith(("ss", "us", "is"))):
        w = w[:-1]
    return w


# P2-6: Unicode tokens. Text is NFKC-normalized (ligatures, full-width
# forms, composed accents) and casefolded (ß → ss, final ς → σ); a token is
# a run of Unicode letters/digits (\w minus "_"), combining marks (so
# Indic vowel signs stay inside their word) and apostrophes. On ASCII text
# this is exactly the old [a-z0-9']+ over text.lower().
def _mark_class():
    out, start = [], None
    for cp in range(0x300, 0x10000):
        is_mark = unicodedata.category(chr(cp)).startswith("M")
        if is_mark and start is None:
            start = cp
        elif not is_mark and start is not None:
            out.append(f"\\u{start:04x}-\\u{cp - 1:04x}" if cp - 1 > start
                       else f"\\u{start:04x}")
            start = None
    return "".join(out)


_TOKEN_RX = re.compile(r"(?:[^\W_]|['" + _mark_class() + r"])+")
_ASCII_TOKEN_RX = re.compile(r"[a-z0-9']+")


def normalize_text(text):
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text, stopwords=None):
    """Index/query tokens. `stopwords`: a collection's list
    (collection_stopwords); default STOPWORDS."""
    sw = STOPWORDS if stopwords is None else stopwords
    if text.isascii():
        # exact fast path: on ASCII, NFKC is the identity, casefold ==
        # lower, and _TOKEN_RX == [a-z0-9']+ (its mark ranges are >= U+0300)
        # — ~3x faster, which keeps uncached index builds at M5-1 speed
        words = _ASCII_TOKEN_RX.findall(text.lower())
    else:
        words = _TOKEN_RX.findall(normalize_text(text))
    return [_light_stem(w) for w in words if w not in sw and len(w) > 1]


def collection_stopwords(col):
    """Stopwords of one collection: STOPWORDS edited by an optional
    <collection>/stopwords.txt — one word per line, '#' comments, a word
    is added, '-word' removes a default (e.g. '-not' to index 'not').
    Words are normalized like text (NFKC + casefold)."""
    sw = set(STOPWORDS)
    root = getattr(col, "root", None)
    if not root:                       # no collection: defaults, no file
        return frozenset(sw)
    path = Path(root) / "stopwords.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    for ln in lines:
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        if ln.startswith("-"):
            sw.discard(normalize_text(ln[1:].strip()))
        else:
            sw.add(normalize_text(ln))
    return frozenset(sw)


# ------------------------------------------------- per-collection settings --
# P2-8: tuning values (TOP_K, MIN_SCORE, MIN_COVERAGE, MIN_DENSE_SIM,
# DENSE_SOLO_SIM, MAX_PER_WORK, POOL_FACTOR, HIERARCHICAL, RERANK,
# TEMPERATURE, QUOTES_TOP_K, QUOTES_TEMPERATURE, QUOTE_MATCH_RATIO, …) are
# read through col.setting(), which falls back to config at CALL time. An
# index built by load_index() carries its collection as `.col`, so functions
# that only get an index (gate_ok, search) are collection-aware too; an
# index built by hand (no `.col`) reads config exactly as before.

def _cfg(obj, key):
    """Setting `key` for a Collection, or for an index carrying `.col`;
    plain config value when neither is available."""
    col = obj if callable(getattr(obj, "setting", None)) \
        else getattr(obj, "col", None)
    if col is not None:
        return col.setting(key, config.get(key))
    return config.get(key)


def _attach_col(index, col):
    """Tag an index and every sub-index it holds with its collection."""
    seen = set()
    stack = [index]
    while stack:
        ix = stack.pop()
        if ix is None or id(ix) in seen:
            continue
        seen.add(id(ix))
        try:
            ix.col = col
        except AttributeError:
            continue
        for attr in ("base", "bm25", "dense"):
            sub = getattr(ix, attr, None)
            if sub is not None and not isinstance(sub, (int, float, str)):
                stack.append(sub)
    return index


def _chunk_files(col):
    """All chunk files of one collection."""
    if not col.chunks_dir.exists():
        return []
    return sorted(col.chunks_dir.glob("*.jsonl"))


def _vector_file(col, key):
    """vectors/<artifact key>.npz, or None."""
    f = col.vectors_dir / (key + ".npz")
    return f if f.exists() else None


def _date_map(col, key):
    """Per-section dates from the stage-1c sidecar (empty when absent)."""
    f = col.dates_dir / (key + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def dedup_ids(chunks, seen=None):
    """Truncated essay titles can collide in chunk ids (same id prefix, same
    40-char title prefix, same offset). Chunk FILES stay untouched (hash
    stability); ids are disambiguated deterministically at read time, in
    file order, by appending #2, #3… Used by every id consumer so the
    reader, quote bank, and sources payload all agree. Pass one `seen`
    dict across files to make ids unique collection-wide (load_sources
    does; ingest.chunk_id_prefixes already keeps sources apart, so this is
    only a safety net)."""
    seen = {} if seen is None else seen
    for c in chunks:
        n = seen.get(c["id"], 0) + 1
        seen[c["id"]] = n
        if n > 1:
            c["id"] = f"{c['id']}#{n}"
    return chunks


def load_sources(col):
    """Returns ordered [(artifact key, chunks, vector_file_or_None)] for one
    collection. Dates and display work-titles are attached in memory only —
    chunk files stay byte-stable.

    Raises errors.NotIndexed when the collection has no chunks at all. A
    chunk file holding zero chunks (e.g. a source too short or empty to
    chunk) is skipped, not returned, so "no chunk files" and "only empty
    chunk files" are treated the same: not indexed. CLIs turn NotIndexed
    into an exit message; the server maps it to HTTP 409."""
    out, seen = [], {}
    for f in _chunk_files(col):
        with open(f) as fh:
            chunks = dedup_ids([json.loads(line) for line in fh
                                if line.strip()], seen)
        if not chunks:
            continue
        key = col.key_of_chunk_file(f)
        dmap = _date_map(col, key)
        for c in chunks:
            # para sidecar lookup by the chunk FILE's key (correct even for
            # a not-yet-migrated schema-1 layout)
            _PARA_KEYS[(col.id, c["source"])] = key
            c["work"] = col.work_title(c["source"])
            if c["essay"] == Path(c["source"]).stem:
                c["essay"] = c["work"]      # untitled section → work title
            d = dmap.get(c["essay"])
            if d:
                c["date"] = d["date"]
                c["date_precision"] = d["precision"]
        out.append((key, chunks, _vector_file(col, key)))
    if not out:
        raise errors.NotIndexed(f"No chunks in collection {col.id!r}. Run: "
                                f".venv/bin/python ingest.py -c {col.id}")
    return out


def load_chunks(col):
    return [c for _, chunks, _ in load_sources(col) for c in chunks]


# ------------------------------------------------------------------- BM25 --

# P2-5: bump when tokenize()/_light_stem() change so on-disk token caches
# (data/cache/tokens/) are rebuilt. The stopword list is fingerprinted into
# the cache key separately (_tokenizer_tag), so editing it also invalidates.
TOKENIZER_VERSION = 2          # 2: Unicode tokenizer (P2-6)


def _tokenizer_tag(stopwords=None):
    import hashlib
    words = STOPWORDS if stopwords is None else stopwords
    sw = hashlib.sha256(" ".join(sorted(words)).encode()).hexdigest()[:16]
    return f"v{TOKENIZER_VERSION}-{sw}"


class BM25Index:
    """Section titles are indexed WITH the chunk text — segmentation strips
    headings from the body, so 'deep sea mining' must still match a chunk
    whose section is titled that even when the body words differ.

    Inverted index (P2-5): `postings[term]` = [(doc idx, tf), …] in doc
    order; a query scores only the docs in its terms' postings. Scores are
    the same float expression, summed in the same (query-token) order as a
    full scan, and ties break the same way ((score, idx) descending), so
    rankings are identical to the pre-P2-5 scan.

    `text_tokens` (optional, aligned with `chunks`): tokenize(c["text"]) per
    chunk, e.g. from the on-disk token cache; the section title is always
    tokenized here (it can change in memory without the chunk file
    changing). tokenize(a + ". " + b) == tokenize(a) + tokenize(b) because
    ". " never belongs to a token."""

    _MEMO_SIZE = 16

    def __init__(self, chunks, text_tokens=None, stopwords=None):
        self.chunks = chunks
        self.stopwords = frozenset(STOPWORDS if stopwords is None
                                   else stopwords)
        if text_tokens is None or len(text_tokens) != len(chunks):
            text_tokens = [self.tokenize(c["text"]) for c in chunks]
        essay_toks = {}
        docs = []
        for c, tt in zip(chunks, text_tokens):
            e = c.get("essay", "")
            et = essay_toks.get(e)
            if et is None:
                et = essay_toks[e] = self.tokenize(e)
            docs.append(et + tt if et else list(tt))
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        n = len(docs)
        self.avg_len = sum(self.doc_len) / max(n, 1)
        postings = {}
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                p = postings.get(t)
                if p is None:
                    postings[t] = [(i, f)]
                else:
                    p.append((i, f))
        self.postings = postings
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in postings.items()}
        self._weights = {}            # term -> [(idx, contribution)], lazy
        self._wkey = None             # (K1, B) the weights were built with
        self._memo = {}               # (query tokens, K1, B) -> [(s, idx)]
        self._lock = threading.Lock()
        self._pos = None              # id(chunk) -> idx, lazy
        self._tf = None

    def tokenize(self, text):
        """tokenize() with this index's stopword list (P2-6)."""
        return tokenize(text, self.stopwords)

    @property
    def tf(self):
        """Per-doc term counts (pre-P2-5 attribute; built on first use)."""
        if self._tf is None:
            self._tf = [Counter(d) for d in self.docs]
        return self._tf

    def _term_weights(self, t, k1, b):
        w = self._weights.get(t)
        if w is None:
            idf = self.idf.get(t, 0)
            dl, avg = self.doc_len, self.avg_len
            # the exact expression of the old full scan, per (doc, term)
            w = [(i, idf * (f * (k1 + 1))
                  / (f + k1 * (1 - b + b * dl[i] / avg)))
                 for i, f in self.postings.get(t, ())]
            self._weights[t] = w
        return w

    def _scored(self, q):
        """All (score, idx) with score > 0 for query tokens `q` (unsorted).
        Memoized per query so the gate and the retrieval that follows it
        (ask_stream; HybridIndex's BM25 leg) score a question only once."""
        k1, b = config.BM25_K1, config.BM25_B
        key = (tuple(q), k1, b)
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None:
                return hit
            if self._wkey != (k1, b):
                self._weights, self._wkey = {}, (k1, b)
            acc = {}
            for t in q:                    # query order, duplicates too
                for i, v in self._term_weights(t, k1, b):
                    acc[i] = acc.get(i, 0.0) + v
            scored = [(s, i) for i, s in acc.items() if s > 0]
            if len(self._memo) >= self._MEMO_SIZE:
                self._memo.pop(next(iter(self._memo)))
            self._memo[key] = scored
            return scored

    def search_idx(self, query, k=None):
        """[(score, doc idx)] best-first; same ranking as search()."""
        k = _cfg(self, "TOP_K") if k is None else k
        scored = self._scored(self.tokenize(query))
        if k is None or k >= len(scored):
            return sorted(scored, reverse=True)
        if k <= 0:                      # old scores[:k] slice semantics
            return sorted(scored, reverse=True)[:k]
        return heapq.nlargest(k, scored)

    def search(self, query, k=None):
        return [(s, self.chunks[i]) for s, i in self.search_idx(query, k)]

    def doc_tokens_of(self, chunk):
        """Indexed tokens (title + text) of one of this index's chunks, or
        None when `chunk` is not one of them."""
        if self._pos is None:
            self._pos = {id(c): i for i, c in enumerate(self.chunks)}
        i = self._pos.get(id(chunk))
        return None if i is None or self.chunks[i] is not chunk \
            else self.docs[i]


# --------------------------------------------------------- token cache --
# Derived data only: <col.data_dir>/cache/tokens/<artifact key>.json, one
# file per chunk file, holding tokenize(c["text"]) for each chunk. Keyed by
# a sha256 of exactly what is tokenized (the chunk texts of that file, in
# order) + the tokenizer tag; any mismatch, parse error or count mismatch
# → rebuild silently. Files whose key has no chunk file anymore are removed
# at load time (ingest's prune only knows library.DERIVED_FILES).

def _token_cache_dir(col):
    return Path(col.data_dir) / "cache" / "tokens"


def _texts_hash(chunks):
    import hashlib
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def _read_token_cache(path, digest, n, tag=None):
    tag = _tokenizer_tag() if tag is None else tag
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("tag") != tag or d.get("hash") != digest:
            return None
        lens, docs = d["lens"], d["docs"]
        if len(lens) != n or len(docs) != n:
            return None
        out = []
        for ln, s in zip(lens, docs):
            toks = s.split(" ") if ln else []
            if len(toks) != ln:
                return None
            out.append(toks)
        return out
    except Exception:
        return None


def _write_token_cache(path, digest, toks, tag=None):
    import os
    tag = _tokenizer_tag() if tag is None else tag
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}."
                             f"{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "hash": digest,
                       "lens": [len(t) for t in toks],
                       "docs": [" ".join(t) for t in toks]}, fh)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass


def cached_text_tokens(col, sources, stopwords=None):
    """tokenize(c["text"], stopwords) for every chunk of `sources`
    (load_sources order), from the per-chunk-file cache where it is still
    valid (the cache tag fingerprints the stopword list)."""
    cdir = _token_cache_dir(col)
    tag = _tokenizer_tag(stopwords)
    out, live = [], set()
    for key, chunks, _ in sources:
        live.add(key + ".json")
        path = cdir / (key + ".json")
        digest = _texts_hash(chunks)
        toks = _read_token_cache(path, digest, len(chunks), tag)
        if toks is None:
            toks = [tokenize(c["text"], stopwords) for c in chunks]
            _write_token_cache(path, digest, toks, tag)
        out.extend(toks)
    try:
        for p in cdir.iterdir():
            if p.is_file() and p.name not in live and p.suffix == ".json":
                p.unlink()
    except Exception:
        pass
    return out


# --------------------------------------------------------- query encoders --
# The encoder lives in encoder.py and is shared with ingest's embed stage
# (same instance, same pooling/max_len). Old names kept as aliases.

OnnxEncoder = encoder.OnnxEncoder


def _query_encoder():
    """Compatibility alias for encoder.get_encoder()."""
    return encoder.get_encoder()


def _check_vectors(f, label, n):
    """Raise ValueError when the loaded .npz `f` can't serve `n` items
    with the current encoder (model/pooling/max_len tag or count)."""
    why = encoder.tag_mismatch(f)
    if why:
        raise ValueError(f"{label}: vectors {why} — re-run ingest.py --force")
    if len(f["vectors"]) != n:
        raise ValueError(f"{label}: vectors/chunks mismatch — "
                         "re-run ingest.py --force")


# ------------------------------------------------------------------ dense --

class DenseIndex:
    """Dense retrieval over every source that has precomputed vectors.
    Sources without vectors are simply not covered by this index (the
    hybrid's BM25 side still sees them)."""

    def __init__(self, sources):
        import numpy as np
        self.np = np
        self.chunks, mats = [], []
        for stem, chunks, vec_file in sources:
            if vec_file is None:
                continue
            f = np.load(vec_file)
            _check_vectors(f, stem, len(chunks))
            vecs = f["vectors"]
            self.chunks.extend(chunks)
            mats.append(vecs)
        if not mats:
            raise ValueError("no vector files found")
        self.vecs = np.vstack(mats)
        self.model = encoder.get_encoder()
        if self.model.dim and self.vecs.shape[1] != self.model.dim:
            raise ValueError("vector dimension differs from the encoder's "
                             "— re-run ingest.py --force")

    def search(self, query, k=None):
        k = _cfg(self, "TOP_K") if k is None else k
        q = self.model.encode([config.EMBED_QUERY_PREFIX + query],
                              normalize_embeddings=True)[0]
        sims = self.vecs @ q
        top = sims.argsort()[::-1][:k]
        # raw cosine (normalized vectors); the gate compares it against
        # MIN_DENSE_SIM / DENSE_SOLO_SIM, fusion only uses ranks
        return [(float(sims[i]), self.chunks[i]) for i in top if sims[i] > 0]


# ----------------------------------------------------------------- hybrid --

class HybridIndex:
    """BM25 + dense, fused with reciprocal-rank fusion:
    fused(c) = Σ_retrievers 1 / (RRF_K + rank_of_c). Robust to the two
    retrievers' incomparable score scales."""

    def __init__(self, sources, bm25=None):
        self.chunks = [c for _, chunks, _ in sources for c in chunks]
        self.dense = DenseIndex(sources)
        # `bm25`: a prebuilt BM25Index over the same chunks (load_index
        # passes one built from the token cache)
        self.bm25 = bm25 if bm25 is not None else BM25Index(self.chunks)

    def search(self, query, k=None):
        k = _cfg(self, "TOP_K") if k is None else k
        n = max(config.HYBRID_CANDIDATES, k)
        fused = {}
        for hits in (self.bm25.search(query, n), self.dense.search(query, n)):
            for rank, (_, c) in enumerate(hits):
                e = fused.setdefault(c["id"], [0.0, c])
                e[0] += 1.0 / (config.RRF_K + rank + 1)
        ranked = sorted(fused.values(), key=lambda e: e[0], reverse=True)
        return [(score, c) for score, c in ranked[:k]]


# -------------------------------------- summaries-first (hierarchical) ----

def load_summaries(col):
    """All Stage-1b summary artifacts:
    [(artifact key, [entries], vec_file_or_None)]."""
    out = []
    if not col.summaries_dir.exists():
        return out
    for f in sorted(col.summaries_dir.glob("*.jsonl")):
        entries = [json.loads(l) for l in open(f)]
        if not entries:
            continue
        key = col.key_of_chunk_file(f)          # summaries/<key>.jsonl
        cand = col.vectors_dir / (key + ".summaries.npz")
        out.append((key, entries, cand if cand.exists() else None))
    return out


class SummaryIndex:
    """Article-level 'explore' stage: which articles are about this query?
    BM25 over summary+keywords+title, RRF-fused with dense summary vectors
    when they exist and an encoder is available."""

    def __init__(self, summaries, stopwords=None):
        self.entries = [e for _, es, _ in summaries for e in es]
        self._texts = [f"{e['essay']}. {e['summary']} {e['keywords']}"
                       for e in self.entries]
        self.bm25 = BM25Index([{"text": t, **e} for t, e in
                               zip(self._texts, self.entries)],
                              stopwords=stopwords)
        self.vecs = self.model = None
        try:
            import numpy as np
            mats, ok = [], True
            for _, es, vec in summaries:
                if vec is None:
                    ok = False
                    break
                f = np.load(vec)
                try:
                    _check_vectors(f, "summaries", len(es))
                except ValueError:
                    ok = False
                    break
                mats.append(f["vectors"])
            if ok and mats:
                model = encoder.get_encoder()   # may raise → both stay None
                self.vecs = np.vstack(mats)
                self.model = model
        except Exception:
            self.vecs = self.model = None    # dense explore off, BM25 remains

    def explore(self, query, m=None):
        """Top-M article keys {(source, essay)} for the query."""
        m = m or config.SUMMARY_TOP_M
        fused = {}
        rankings = [self.bm25.search(query, m * 3)]
        if self.vecs is not None and self.model is not None:
            q = self.model.encode([config.EMBED_QUERY_PREFIX + query],
                                  normalize_embeddings=True)[0]
            sims = self.vecs @ q
            top = sims.argsort()[::-1][:m * 3]
            rankings.append([(float(sims[i]), self.entries[i]) for i in top])
        for hits in rankings:
            for rank, (_, e) in enumerate(hits):
                key = (e["source"], e["essay"])
                fused[key] = fused.get(key, 0) + 1 / (config.RRF_K + rank + 1)
        return set(sorted(fused, key=fused.get, reverse=True)[:m])


class HierarchicalIndex:
    """Two-stage retrieval: explore article summaries, then boost chunks
    from the selected articles inside the usual chunk-level ranking."""

    def __init__(self, base, summary_index):
        self.base = base
        self.summaries = summary_index
        self.chunks = base.chunks
        self.bm25 = base if isinstance(base, BM25Index) else getattr(base, "bm25", None)
        self.dense = base if isinstance(base, DenseIndex) else getattr(base, "dense", None)

    def search(self, query, k=None):
        k = _cfg(self, "TOP_K") if k is None else k
        selected = self.summaries.explore(query)
        pool = self.base.search(query, k * 2)
        scored = []
        for rank, (_, c) in enumerate(pool):
            w = config.RESTRICTED_WEIGHT if (c["source"], c["essay"]) in selected else 1.0
            scored.append((w / (config.RRF_K + rank + 1), c))
        scored.sort(key=lambda e: e[0], reverse=True)
        return scored[:k]


def load_index(col):
    sources = load_sources(col)
    chunks = [c for _, cs, _ in sources for c in cs]
    mode = col.setting("RETRIEVAL", config.RETRIEVAL)
    have_vectors = any(v for _, _, v in sources)
    _bm = []

    sw = collection_stopwords(col)

    def bm25():                       # built once, from the token cache
        if not _bm:
            _bm.append(BM25Index(chunks, cached_text_tokens(col, sources, sw),
                                 stopwords=sw))
        return _bm[0]

    if mode == "bm25" or (mode == "auto" and not have_vectors):
        base = bm25()
    else:
        try:
            if mode == "dense":
                cls = DenseIndex
                base = DenseIndex(sources)
            else:                                   # auto/hybrid → hybrid
                cls = HybridIndex
                base = HybridIndex(sources, bm25=bm25())
        except ImportError:
            print("[rag] vectors found but no query encoder available "
                  "— using BM25. (bash setup.sh, or bundle the ONNX encoder)")
            base = bm25()
        except Exception as e:
            print(f"[rag] {cls.__name__} unavailable ({e}) — using BM25. "
                  f"Re-run: .venv/bin/python ingest.py -c {col.id} --force")
            base = bm25()

    hier = col.setting("HIERARCHICAL", config.HIERARCHICAL)
    if hier in ("auto", "on"):
        summaries = load_summaries(col)
        if summaries:
            try:
                return _attach_col(
                    HierarchicalIndex(base, SummaryIndex(summaries,
                                                         stopwords=sw)), col)
            except Exception as e:
                print(f"[rag] hierarchical retrieval unavailable ({e}) — flat")
        elif hier == "on":
            print("[rag] HIERARCHICAL=on but no summaries — run summarize.py")
    return _attach_col(base, col)


_RERANKER = None


def _maybe_rerank(query, hits, mode=None):
    """Optional cross-encoder reordering of the candidate pool (S4b).
    `mode` is the collection's RERANK setting (default: config.RERANK)."""
    global _RERANKER
    mode = config.RERANK if mode is None else mode
    if mode == "off" or not hits:
        return hits
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
            _RERANKER = CrossEncoder(config.RERANK_MODEL)
        except Exception:
            _RERANKER = False               # unavailable; don't retry
    if not _RERANKER:
        return hits
    scores = _RERANKER.predict([(query, c["text"]) for _, c in hits])
    return sorted(((float(s), c) for s, (_, c) in zip(scores, hits)),
                  key=lambda e: e[0], reverse=True)


def diversify(hits, k, cap=None):
    """Cap chunks per work within the top-k so one large volume can't crowd
    out the rest of the corpus; backfill by rank if the cap leaves gaps."""
    cap = cap or config.MAX_PER_WORK
    picked, spill, counts = [], [], Counter()
    for s, c in hits:
        if len(picked) == k:
            break
        if counts[c["source"]] < cap:
            picked.append((s, c))
            counts[c["source"]] += 1
        else:
            spill.append((s, c))
    for h in spill:
        if len(picked) >= k:
            break
        picked.append(h)
    return picked[:k]


# ------------------------------------------------------------------- gate --

def _gate_min_score(n_docs, min_score=None):
    """MIN_SCORE was tuned on a ~5k-chunk corpus; BM25 scores scale roughly
    with log(corpus size) via the idf term, so scale the gate down for small
    collections instead of blocking everything they could answer."""
    min_score = config.MIN_SCORE if min_score is None else min_score
    return min_score * min(1.0, math.log1p(n_docs) / math.log1p(5000))


def bm25_gate_ok(bm, question, with_hits=False, settings=None, hits=None):
    """BM25 signal: top score >= the log-scaled MIN_SCORE and a top-k hit
    covering >= MIN_COVERAGE of the query terms. `settings` (a Collection
    or an index carrying `.col`) supplies the thresholds; default: `bm`.

    `hits` (P2-5): the caller's own `bm.search(question, k)` result with
    k >= TOP_K (or k=None for the full ranking) — used instead of searching
    again; its first TOP_K entries are exactly what the search here would
    return. Without it, bm is searched as before."""
    src = bm if settings is None else settings
    top_k = _cfg(src, "TOP_K")
    hits = bm.search(question, top_k) if hits is None else list(hits)[:top_k]
    tok = getattr(bm, "tokenize", tokenize)     # the index's stopwords
    qtoks = set(tok(question))
    tok_of = getattr(bm, "doc_tokens_of", None)

    def doc_toks(c):
        d = tok_of(c) if tok_of else None
        return set(tok(c.get("essay", "") + " " + c["text"])) \
            if d is None else set(d)
    cov = max((sum(1 for t in qtoks if t in dt) / len(qtoks)
               for dt in (doc_toks(c) for _, c in hits)),
              default=0) if qtoks else 0
    ok = bool(hits and hits[0][0] >= _gate_min_score(
                  len(bm.chunks), _cfg(src, "MIN_SCORE"))
              and cov >= _cfg(src, "MIN_COVERAGE"))
    return (ok, hits) if with_hits else ok


def dense_top_sim(dn, question):
    """Raw cosine of the best-matching chunk (0.0 when nothing is > 0)."""
    hits = dn.search(question, 1)
    return hits[0][0] if hits else 0.0


def gate_ok(index, question, bm25_hits=None):
    """Is the question answerable from the corpus?

    `bm25_hits` (optional): the caller's `index.bm25.search(question, k)`
    result, k >= TOP_K — reused instead of a second BM25 search. (Repeat
    searches of one question are also memoized inside BM25Index.)

    BM25-only index: the BM25 signal alone (unchanged behaviour).
    With a dense signal (cosine of the top chunk), bge puts unrelated text
    at ~0.3-0.64, so dense can't pass by a low bar and BM25 can't pass on
    its own either (it false-accepts on shared words):
        cos >= DENSE_SOLO_SIM                  — dense alone is convincing
        or (BM25 passes and cos >= MIN_DENSE_SIM) — both signals agree
    Thresholds and the measurements behind them: config.py; each one is
    per-collection (P2-8) through the index's `.col`."""
    dn = index if isinstance(index, DenseIndex) else getattr(index, "dense", None)
    bm = index if isinstance(index, BM25Index) else getattr(index, "bm25", None)
    if dn is None:
        return bool(bm) and bm25_gate_ok(bm, question, settings=index,
                                         hits=bm25_hits)
    if bm is None:                  # dense-only retrieval: still gate on both
        bm25_hits = None            # caller's hits can't be from this index
        bm = getattr(dn, "_gate_bm25", None)
        if bm is None:              # with the collection's stopwords (P2-6)
            col = getattr(index, "col", None) or getattr(dn, "col", None)
            bm = dn._gate_bm25 = BM25Index(
                dn.chunks, stopwords=collection_stopwords(col))
    cos = dense_top_sim(dn, question)
    if cos >= _cfg(index, "DENSE_SOLO_SIM"):
        return True
    ok, hits = bm25_gate_ok(bm, question, with_hits=True, settings=index,
                            hits=bm25_hits)
    if not ok:
        return False
    if cos >= _cfg(index, "MIN_DENSE_SIM"):
        return True
    # a source without vectors has no dense signal: there BM25 still decides
    covered = getattr(dn, "_gate_ids", None)
    if covered is None:
        covered = dn._gate_ids = {c["id"] for c in dn.chunks}
    return hits[0][1]["id"] not in covered


# -------------------------------------------------------------- generation --

def _request(url, payload):
    return urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})


_EMBEDDED_LLM = None
# One shared in-process Llama serves every server thread; llama.cpp contexts
# are not re-entrant, so a generation holds this for its whole stream.
_EMBEDDED_LOCK = threading.Lock()


_EMBEDDED_KEY = None     # (model path, ctx, threads) the loaded GGUF used


def reset_backend():
    """Forget the loaded embedded model so the next generation loads it with
    the current settings (serve.py calls this after a settings change). An
    in-flight generation keeps its own reference and finishes normally."""
    global _EMBEDDED_LLM, _EMBEDDED_KEY
    _EMBEDDED_LLM = None
    _EMBEDDED_KEY = None


def _embedded_llm():
    """Lazy-load the bundled GGUF once (first question pays the load cost);
    reloaded when LLM_MODEL_PATH / LLM_CTX / LLM_THREADS change."""
    global _EMBEDDED_LLM, _EMBEDDED_KEY
    want = (str(config.get("LLM_MODEL_PATH")), config.get("LLM_CTX"),
            config.get("LLM_THREADS"))
    if _EMBEDDED_KEY is not None and _EMBEDDED_KEY != want:
        _EMBEDDED_LLM = None
    if _EMBEDDED_LLM is None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "LLM_BACKEND is 'embedded' but llama-cpp-python is not "
                "installed (pip install llama-cpp-python).")
        if not Path(want[0]).exists():
            raise RuntimeError(f"Model file missing: {want[0]}")
        kwargs = {"model_path": want[0], "n_ctx": want[1], "verbose": False}
        if want[2]:
            kwargs["n_threads"] = want[2]
        _EMBEDDED_LLM = Llama(**kwargs)
        _EMBEDDED_KEY = want
    return _EMBEDDED_LLM


def _open(req, model):
    """urlopen that surfaces the backend's actual error text (e.g. Ollama's
    'model not found') instead of a generic connection message."""
    try:
        return urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = str(e)
        raise RuntimeError(f"{config.get('LLM_BACKEND')} backend error for model "
                           f"'{model}': {detail}") from None


def llm_stream(system, user, temperature=None, model=None, backend=None,
               url=None, max_tokens=None):
    """Yield text deltas from the configured backend (config.LLM_BACKEND).
    `model`/`backend`/`url` override the defaults for this call (used by
    persona mode's dedicated voice model/server); `max_tokens` overrides
    config.LLM_MAX_TOKENS (the follow-up rewrite uses a small cap)."""
    # every value read NOW through config.get (settings apply without restart)
    temperature = config.get("TEMPERATURE") if temperature is None \
        else temperature
    max_tokens = config.get("LLM_MAX_TOKENS") if max_tokens is None \
        else max_tokens
    model = model or config.get("LLM_MODEL")
    backend = backend or config.get("LLM_BACKEND")
    url = url or config.get("LLM_URL")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if backend == "ollama":
        req = _request(f"{url}/api/chat", {
            "model": model, "messages": messages, "stream": True,
            "options": {"temperature": temperature,
                        "num_ctx": config.get("LLM_CTX"),
                        "num_predict": max_tokens}})
        with _open(req, model) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = obj.get("message", {}).get("content", "")
                if t:
                    yield t
                if obj.get("done"):
                    break
    elif backend == "openai":
        base = url.rstrip("/")
        req = _request(f"{base}/chat/completions", {
            "model": model, "messages": messages,
            "temperature": temperature, "stream": True,
            "max_tokens": max_tokens})
        with _open(req, model) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = (json.loads(data)["choices"][0]
                         .get("delta", {}).get("content"))
                if delta:
                    yield delta
    elif backend == "embedded":
        # Held for the full stream (model load included). Released in
        # `finally`, so a generator that is closed early or garbage-collected
        # unlocks too.
        _EMBEDDED_LOCK.acquire()
        try:
            llm = _embedded_llm()
            parts = llm.create_chat_completion(
                messages=messages, temperature=temperature, stream=True,
                max_tokens=max_tokens)
            try:
                for part in parts:
                    delta = part["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
            finally:
                close = getattr(parts, "close", None)
                if close:
                    close()
        finally:
            _EMBEDDED_LOCK.release()
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}")


def llm_chat(system, user):
    """Blocking variant."""
    return "".join(llm_stream(system, user))


# ---------------------------------------------------------------- prompts --
# Templated per collection: {label} = what the corpus is (name + optional
# description/author), {author} = named author (persona/quotes flavor).

def _collection_label(col):
    label = f'the document collection "{col.name}"'
    if col.author:
        label += f" by {col.author}"
    if col.description:
        label += f" ({col.description})"
    return label


def persona_disclaimer(col):
    return (f"*[AI reconstruction from the collected writings of "
            f"{col.author} — not the author.]*\n\n")


# Scholar and persona excerpts are numbered [1]…[n] (P2-2): small models
# follow short numeric markers far more reliably than long titles, and the
# markers can be checked mechanically (check_citations). Quotes mode keeps
# its [Section Title, p. N] citations and its verbatim verification.
_CITE_RULE = ("The excerpts are numbered [1], [2], … . Cite every claim with "
              "the number of the excerpt it comes from in square brackets, "
              "e.g. [1] or [2][3]. Use only the numbers of the excerpts "
              "given.")


def system_scholar(col):
    return f"""You are a careful research assistant answering questions about \
{_collection_label(col)}. Answer ONLY from the excerpts provided. Quote short \
passages where useful. {_CITE_RULE} Excerpt headers may include the year the piece was written; \
when the question concerns how views changed or developed over time, organize the \
answer chronologically and name the years. If the excerpts do not answer the \
question, say the documents provided do not cover it. Never invent facts."""


def system_persona(col):
    return f"""You speak in the first person as {col.author}, in the direct \
style of the source texts. Ground every statement strictly in the excerpts \
provided from their writings; do not invent positions. If the excerpts do not \
cover the question, say plainly that your writings here do not address it. Keep \
the voice consistent with the excerpts. {_CITE_RULE}"""


# When a fine-tuned voice model is active the style lives in the weights;
# only the grounding contract remains in the prompt.
def system_persona_tuned(col):
    return f"""You are {col.author}. Ground every statement strictly in the \
excerpts provided from your writings; do not invent positions. If the excerpts \
do not cover the question, say plainly that your writings here do not address it. \
{_CITE_RULE}"""


def system_quotes(col):
    return f"""You are a careful anthologist of {_collection_label(col)}. Answer \
using ONLY verbatim quotations from the excerpts provided, arranged so they read \
as one cohesive whole. Rules: begin with a single short framing sentence in your \
own neutral voice; then give 3-6 quotations as Markdown blockquotes (lines starting \
with '> '); copy each quotation EXACTLY, word for word, from the excerpts — never \
alter, blend, or shorten text except with an ellipsis (…); end every blockquote \
with its citation in the form [Section Title, p. N]; between quotations you may add \
at most one short connective line in the curator's voice, never presented as the \
source's words. Order the quotations to build the argument. If the excerpts do not \
answer the question, say so instead of quoting loosely."""


def not_covered(col):
    return (f'The documents in "{col.name}" do not appear to cover that. '
            f"Try rephrasing, or ask about what the collection contains.")


MODES = {
    "scholar": (system_scholar, None, None),
    "persona": (system_persona, None, None),
    "quotes": (system_quotes, "QUOTES_TEMPERATURE", "QUOTES_TOP_K"),
}


def available_modes(col):
    """Modes this collection supports (persona is opt-in + needs an author)."""
    return ["scholar", "quotes"] + (["persona"] if col.persona else [])


def chunk_year(c):
    """Display/filter year for a chunk, or None (work-precision excluded)."""
    if c.get("date") and c.get("date_precision") != "work":
        return int(c["date"][:4])
    return None


def build_context(hits, numbered=True):
    """Excerpts joined by '---'. numbered=True (scholar/persona) prefixes
    excerpt i (1-based, in the given order) with "[i] "; quotes mode passes
    numbered=False and keeps the plain [label, pages] headers."""
    parts = []
    for i, (_, c) in enumerate(hits, start=1):
        pages = (f"p. {c['page_start']}" if c["page_start"] == c["page_end"]
                 else f"pp. {c['page_start']}-{c['page_end']}")
        work = c.get("work", Path(c["source"]).stem)
        label = c["essay"] if c["essay"] == work else f"{c['essay']} — {work}"
        y = chunk_year(c)
        year = f", {y}" if y else ""
        if numbered:
            # "[1] Title, p. 3" — no nested brackets: small models copied
            # "[1] [Title, p. 3]" headers into answers instead of citing
            parts.append(f"[{i}] {label}{year}, {pages}\n{c['text']}")
        else:
            parts.append(f"[{label}{year}, {pages}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# -------------------------------------------------------------- citations --

# A bracket holding ONLY numbers, commas, spaces and ranges: [3], [1, 3],
# [1-3], [1–3]. Anything with letters ([Section, p. 3]) is not a marker.
_CITE_RX = re.compile(r"\[\s*(\d+(?:\s*(?:,|-|–|—)\s*\d+)*)\s*\]")
_MAX_RANGE = 50       # [1-3000] is a typo or a year span, not a citation run


def citation_markers(answer):
    """Every number cited by a numeric [n] marker in `answer`, in order of
    appearance (repeats kept). [2][3] is two markers; [1-3] expands."""
    out = []
    for m in _CITE_RX.finditer(answer or ""):
        for part in m.group(1).split(","):
            ends = [int(x) for x in re.split(r"\s*[-–—]\s*", part.strip())
                    if x]
            if len(ends) == 2 and 0 <= ends[1] - ends[0] <= _MAX_RANGE:
                out.extend(range(ends[0], ends[1] + 1))
            else:
                out.extend(ends)
    return out


def check_citations(answer, n):
    """Map [n] markers in `answer` against n numbered excerpts. Returns
    {"used", "unused", "out_of_range"} — sorted, de-duplicated ints:
    used = cited numbers within 1..n, unused = 1..n never cited,
    out_of_range = cited numbers outside 1..n."""
    nums = set(citation_markers(answer))
    used = sorted(x for x in nums if 1 <= x <= n)
    return {"used": used,
            "unused": [i for i in range(1, n + 1) if i not in nums],
            "out_of_range": sorted(x for x in nums if not 1 <= x <= n)}


# ------------------------------------------------------------- follow-ups --
# P2-1. A follow-up ("what about after 2020?", "why did they do that?")
# retrieves nothing useful on its own, so it is rewritten into a standalone
# question with ONE short LLM call before retrieval and the gate.
#
# is_followup() is a deterministic heuristic, applied only when there IS
# recent history (a first turn is never rewritten). A question is a
# follow-up when ANY of these holds:
#   1. pronoun/demonstrative: it, its, they, them, their, this, that, these,
#      those, he, him, his, she, her, hers, the same, such, the former,
#      the latter, there
#   2. continuation / ellipsis cue at the start: and, but, or, also, so,
#      then, what about, how about, what else, why not, more, tell me more,
#      go on, continue, same, compared; or an explicit "..." / "…"
#   3. short: at most FOLLOWUP_MAX_TERMS content terms (after stopwords).
# A false positive costs one small LLM call whose rewrite of an already
# standalone question is that question; a miss just answers as before.

_PRONOUN_RX = re.compile(
    r"\b(?:it|its|they|them|their|theirs|this|that|these|those|he|him|his|"
    r"she|her|hers|there|such|the same|the former|the latter)\b", re.I)
_CONTINUE_RX = re.compile(
    r"^\s*(?:and|but|or|also|so|then|what about|how about|what else|"
    r"why not|more|tell me more|go on|continue|same|compared)\b", re.I)
FOLLOWUP_MAX_TERMS = 2


def is_followup(question):
    """True when `question` looks like it depends on earlier turns (see the
    rules above). Pure function of the text."""
    q = question or ""
    if "..." in q or "…" in q:
        return True
    if _PRONOUN_RX.search(q) or _CONTINUE_RX.search(q):
        return True
    return len(tokenize(q)) <= FOLLOWUP_MAX_TERMS


def _clip_words(text, n):
    """Whitespace-collapsed text, cut to n words (+ "…" when cut)."""
    w = (text or "").split()
    return " ".join(w[:n] + (["…"] if len(w) > n else []))


def _history_text(turns, answer_words):
    lines = []
    for t in turns:
        lines.append(f"Q: {_clip_words(t.get('question', ''), answer_words)}")
        lines.append(f"A: {_clip_words(t.get('answer', ''), answer_words)}")
    return "\n".join(lines)


REWRITE_SYSTEM = ("You rewrite a follow-up question into one standalone "
                  "question that can be understood without the conversation. "
                  "Resolve pronouns and references using the conversation. "
                  "Do not answer it. Reply with the standalone question only, "
                  "on one line.")


def _clean_rewrite(text, question):
    """The rewrite's first non-empty line, stripped of labels/quotes, or
    None when it is junk (empty, no letters, or implausibly long)."""
    line = next((l.strip() for l in (text or "").splitlines() if l.strip()),
                "")
    line = re.sub(r"^(?:standalone(?: question)?|question|rewritten"
                  r"(?: question)?)\s*:\s*", "", line, flags=re.I)
    line = line.strip().strip('"“”\'`').strip()
    if not re.search(r"[A-Za-z]", line):
        return None
    limit = max(40, 3 * len(question.split()) + 30)
    if len(line.split()) > limit or len(line) > 400:
        return None
    return line


def rewrite_question(col, question, history):
    """One standalone-question LLM call (temperature 0, small max_tokens).
    Returns the rewrite, or None when there is no history, the question is
    not a follow-up, the call fails, or the reply is junk (the caller then
    keeps the original question)."""
    if not history or not is_followup(question):
        return None
    words = _cfg(col, "HISTORY_ANSWER_WORDS")
    user = (f"Conversation:\n{_history_text(history, words)}\n\n"
            f"Follow-up question: {question}\n\nStandalone question:")
    try:
        text = "".join(llm_stream(REWRITE_SYSTEM, user, temperature=0,
                                  max_tokens=_cfg(col, "REWRITE_MAX_TOKENS")))
    except Exception:
        return None
    out = _clean_rewrite(text, question)
    if out is None or out.strip().lower() == question.strip().lower():
        return None
    return out


# ------------------------------------------------------------ token budget --
# Estimated, not tokenized: words x 1.35 over every message sent. Integer
# form (words x 135 vs. tokens x 100) avoids float edge cases at the limit.

_TOK_PER_WORD_X100 = 135


def _words(text):
    return len(text.split())


# Last line of numbered (scholar/persona) prompts. Small models (1.5B)
# ignored the system prompt's citation rule, or answered with a bare "[1]";
# a short reminder with a worked example right before the answer fixes most
# of both (measured: see plan.md, "Citation prompt").
CITE_REMINDER = ("Answer in your own words in a few sentences. After each "
                 "sentence, put the number of the excerpt it is based on in "
                 "square brackets, like this: The bridge opened in 1932 [2]. "
                 "Never reply with only a number.")


def user_prompt(context, question, history_text="", standalone=None,
                numbered=False):
    """User message: optional recent conversation, the excerpts, and the
    question (with its standalone rewrite when there is one); numbered
    prompts end with CITE_REMINDER."""
    head = (f"Earlier in this conversation:\n{history_text}\n\n"
            if history_text else "")
    q = f"{question}\n(Standalone form: {standalone})" if standalone \
        else question
    tail = f"\n\n{CITE_REMINDER}" if numbered and context else ""
    return (f"{head}Excerpts from the documents:\n\n{context}\n\n"
            f"Question: {q}{tail}")


def _order_for_prompt(hits, question):
    """Evolution questions present excerpts chronologically; otherwise the
    retrieval rank order is kept."""
    if _EVOLUTION.search(question):
        return sorted(hits, key=lambda h: (chunk_year(h[1]) is None,
                                           chunk_year(h[1]) or 0))
    return list(hits)


def fit_context(system, question, hits, history=None, numbered=True,
                standalone=None, order_question=None):
    """Budget the prompt to the model window.

    `hits` must be ranked best-first. Returns (sent_hits, user, dropped):
    the lowest-ranked hits are removed until
    words(system + user) x 1.35 + LLM_MAX_TOKENS <= LLM_CTX. If even the
    best hit alone does not fit, its text is truncated (on a copy; the
    indexed chunk is never mutated, and the copy carries "truncated": True)
    rather than sending no excerpt. sent_hits are in presentation order
    (chronological for evolution questions). Config is read at call time.

    `history` (recent turns, oldest first) is trimmed BEFORE any excerpt:
    oldest turns go first, and only once no history is left are hits
    dropped. See fit_prompt() for how many turns were kept."""
    sent, user, dropped, _ = fit_prompt(system, question, hits, history,
                                        numbered, standalone, order_question)
    return sent, user, dropped


def fit_prompt(system, question, hits, history=None, numbered=True,
               standalone=None, order_question=None, answer_words=None):
    """fit_context() plus the number of history turns kept:
    (sent_hits, user, dropped, history_used)."""
    budget_x100 = (config.LLM_CTX - config.LLM_MAX_TOKENS) * 100
    sys_words = _words(system)
    history = list(history or [])
    answer_words = (config.HISTORY_ANSWER_WORDS if answer_words is None
                    else answer_words)
    oq = order_question or question

    def build(kept, turns=()):
        ordered = _order_for_prompt(kept, oq)
        user = user_prompt(build_context(ordered, numbered), question,
                           _history_text(turns, answer_words), standalone,
                           numbered=numbered)
        ok = (sys_words + _words(user)) * _TOK_PER_WORD_X100 <= budget_x100
        return ok, ordered, user

    # history first: keep the most recent turns that fit with ALL hits
    if hits:
        for n_hist in range(len(history), 0, -1):
            ok, ordered, user = build(list(hits), history[-n_hist:])
            if ok:
                return ordered, user, 0, n_hist

    kept = list(hits)
    while kept:
        ok, ordered, user = build(kept)
        if ok:
            return ordered, user, len(hits) - len(kept), 0
        if len(kept) == 1:
            break
        kept.pop()                    # lowest retrieval rank goes first
    if not kept:
        return [], user_prompt("", question, standalone=standalone), 0, 0

    # A single excerpt is still over budget: truncate its text to fit.
    s, c = kept[0]
    text_words = c["text"].split()
    _, _, user = build(kept)
    overhead = sys_words + _words(user) - len(text_words)
    allowed = max(budget_x100 // _TOK_PER_WORD_X100 - overhead - 1, 0)  # "…"
    short = dict(c, text=" ".join(text_words[:allowed] + ["…"]),
                 truncated=True)
    _, ordered, user = build([(s, short)])
    return ordered, user, len(hits) - 1, 0


# ------------------------------------------------------------ era parsing --

_ERA_PATTERNS = [
    (re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", re.I),
     lambda m: (int(m.group(1)), int(m.group(2)))),
    (re.compile(r"\bafter\s+(\d{4})\b", re.I), lambda m: (int(m.group(1)), 9999)),
    (re.compile(r"\bsince\s+(\d{4})\b", re.I), lambda m: (int(m.group(1)), 9999)),
    (re.compile(r"\bbefore\s+(\d{4})\b", re.I), lambda m: (0, int(m.group(1)))),
    (re.compile(r"\buntil\s+(\d{4})\b", re.I), lambda m: (0, int(m.group(1)))),
    (re.compile(r"\bin\s+the\s+(\d{3})0s\b", re.I),
     lambda m: (int(m.group(1)) * 10, int(m.group(1)) * 10 + 9)),
    (re.compile(r"\b(?:in|of|from)\s+(\d{4})\b", re.I),
     lambda m: (int(m.group(1)), int(m.group(1)))),
]

_EVOLUTION = re.compile(r"\bevolv|\bover time\b|\bchange[ds]?\b|\bhistory of\b|"
                        r"\bdevelop(?:ed|ment)\b|\bthrough the (?:years|decades)\b|"
                        r"\bearl(?:y|ier) (?:and|vs|versus) lat(?:e|er)\b", re.I)


def parse_era(question):
    """(lo, hi) year range implied by the question, or None."""
    for rx, f in _ERA_PATTERNS:
        m = rx.search(question)
        if m:
            lo, hi = f(m)
            if 1900 <= lo <= 2100 or 1900 <= hi <= 2100:
                return (lo, hi)
    return None


def apply_era(hits, era, k):
    """Soft-boost hits inside the era; hard-filter only when enough remain."""
    if not era:
        return hits
    lo, hi = era
    inside = [(s, c) for s, c in hits
              if (y := chunk_year(c)) is not None and lo <= y <= hi]
    if len(inside) >= k:
        return inside
    outside = [(s, c) for s, c in hits if (s, c) not in inside]
    return inside + outside          # boost: in-era first, rest as backfill


# ----------------------------------------------------------- source reader --

def _chunk_start(c):
    """Word offset of a chunk in its section: the explicit "start" (P2-4),
    else the last id segment (see ingest.py). Deduped ids carry a '#n'
    suffix (see dedup_ids) — strip it."""
    if isinstance(c.get("start"), int):
        return c["start"]
    return int(c["id"].rsplit("-", 1)[-1].split("#")[0])


def _overlap(prev, ch, step, ov):
    """Words at the head of `ch` that repeat the tail of `prev` (the chunk
    before it in its section), or None when there is a gap between them.
    Chunks with explicit offsets (P2-4: "start"/"end") use them; older
    chunks fall back to the fixed-window arithmetic (start diff == step →
    exactly `ov` words)."""
    ps, pe, cs = prev.get("start"), prev.get("end"), ch.get("start")
    if all(isinstance(x, int) for x in (ps, pe, cs)):
        if cs < ps or cs > pe:
            return None
        return min(pe - cs, len(ch["text"].split()))
    return ov if _chunk_start(ch) - _chunk_start(prev) == step else None


def build_reader(chunks):
    """Returns (by_id, groups): chunk lookup and per-(source, essay) lists
    ordered by word offset, for neighbor navigation."""
    by_id, groups = {}, {}
    for c in chunks:
        by_id[c["id"]] = c
        groups.setdefault((c["source"], c["essay"]), []).append(c)
    for seq in groups.values():
        seq.sort(key=_chunk_start)
    return by_id, groups


_PARA_MAPS = {}
_PARA_KEYS = {}        # (col.id, source filename) -> artifact key


def _paras_for(col, chunk):
    """Display-only paragraph offsets for a chunk (from the para sidecar
    written at chunk time); [] when no map exists — reader then shows the
    passage as a single block, exactly as before."""
    akey = (_PARA_KEYS.get((col.id, chunk["source"]))
            or col.artifact_key(chunk["source"]))
    key = (col.id, akey)
    if key not in _PARA_MAPS:
        m = {}
        f = col.paras_dir / (akey + ".json")
        if f.exists():
            try:
                m = json.loads(f.read_text())
            except Exception:
                m = {}
        _PARA_MAPS[key] = m
    return _PARA_MAPS[key].get(chunk["id"], [])


def _render_words(words, paras):
    """Join words, inserting a paragraph break before each index in paras."""
    out = []
    for i, w in enumerate(words):
        if i in paras and i:
            out.append("\n\n")
        elif i:
            out.append(" ")
        out.append(w)
    return "".join(out)


def passage(col, by_id, groups, chunk_id, before=1, after=1):
    """Continuous text around a cited chunk, with the chunk's exact overlap
    removed at every stitch so nothing repeats (_overlap: the explicit
    start/end offsets, or CHUNK_OVERLAP_WORDS for pre-P2-4 chunks).
    Paragraph breaks (\\n\\n) are restored from the display-only para maps."""
    c = by_id[chunk_id]
    seq = groups[(c["source"], c["essay"])]
    i = next(j for j, ch in enumerate(seq) if ch["id"] == chunk_id)
    chunk_w = col.setting("CHUNK_WORDS", config.CHUNK_WORDS)
    ov = col.setting("CHUNK_OVERLAP_WORDS", config.CHUNK_OVERLAP_WORDS)
    step = chunk_w - ov
    prevs = seq[max(0, i - before):i]
    nexts = seq[i + 1:i + 1 + after]

    def assemble(chain, anchor_prev=None, drop_tail=0):
        """Word-space stitch: returns rendered text with paragraph breaks.
        Trims each chunk head that overlaps its predecessor; a non-adjacent
        gap becomes its own '[…]' paragraph."""
        words, paras = [], set()
        prev = anchor_prev
        for ch in chain:
            w = ch["text"].split()
            skip = 0
            if prev is not None:
                o = _overlap(prev, ch, step, ov)
                if o is not None:
                    skip = o
                else:
                    paras.add(len(words))
                    words.append("[…]")
                    paras.add(len(words))
            base = len(words) - skip
            for p in _paras_for(col, ch):
                if p >= skip:
                    paras.add(base + p)
            words.extend(w[skip:])
            prev = ch
        if drop_tail and words:
            words = words[:-drop_tail] if len(words) > drop_tail else []
            paras = {p for p in paras if p < len(words)}
        return _render_words(words, paras)

    drop, gap = 0, False
    if prevs:
        o = _overlap(prevs[-1], c, step, ov)
        gap = o is None
        drop = o or 0      # before-block tail duplicates the cited chunk head
    before_text = assemble(prevs, drop_tail=drop)
    if gap:
        before_text += " […]"
    cited_words = c["text"].split()
    after_text = assemble(nexts, anchor_prev=c)
    cited_text = _render_words(cited_words, set(_paras_for(col, c)))

    included = prevs + [c] + nexts
    return {"work": c.get("work", Path(c["source"]).stem), "essay": c["essay"],
            "page_start": min(ch["page_start"] for ch in included),
            "page_end": max(ch["page_end"] for ch in included),
            "before": before_text, "cited": cited_text, "after": after_text,
            "has_more_before": i - before > 0,
            "has_more_after": i + 1 + after < len(seq)}


# -------------------------------------------------------------- quote bank --

def load_quote_bank(col):
    """All stage-1d quotable passages of one collection."""
    bank = []
    if col.quotes_dir.exists():
        for f in sorted(col.quotes_dir.glob("*.jsonl")):
            lines = list(open(f))
            bank.extend(json.loads(l) for l in lines[1:] if l.strip())
    return bank


def random_quote(bank, by_id, index=None, topic=None, era=None, rng=None):
    """One score-weighted random bank entry, verified verbatim against its
    chunk before returning. Filters: topic (BM25 over chunks), era (lo, hi)."""
    import random as _random
    rng = rng or _random
    pool = bank
    if topic and index is not None:
        top_ids = {c["id"] for _, c in index.search(topic, 60)}
        pool = [e for e in pool if e["chunk_id"] in top_ids] or pool
    if era:
        lo, hi = era

        def in_era(e):
            c = by_id.get(e["chunk_id"])
            y = chunk_year(c) if c else None
            return y is not None and lo <= y <= hi
        pool = [e for e in pool if in_era(e)] or pool
    weights = [e.get("score", 1.0) for e in pool]
    for _ in range(10):                     # belt and braces: verbatim assert
        e = rng.choices(pool, weights=weights, k=1)[0]
        c = by_id.get(e["chunk_id"])
        if c and e["text"] in c["text"]:
            return {"text": e["text"], "essay": e["essay"],
                    "work": c.get("work", Path(e["source"]).stem),
                    "year": chunk_year(c), "page": e["page"],
                    "chunk_id": e["chunk_id"]}
    return None


# ----------------------------------------------------- quote verification --

def _norm(text):
    """Normalize for quote matching: straight quotes/dashes, one space, lower."""
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", "-").replace("–", "-"))
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_quotes(answer):
    """Markdown blockquotes from an answer, trailing [citation] stripped.
    Consecutive '>' lines form one quote."""
    quotes, buf = [], []
    for line in answer.splitlines() + [""]:
        if line.lstrip().startswith(">"):
            buf.append(line.lstrip().lstrip(">").strip())
        elif buf:
            q = " ".join(buf).strip()
            q = re.sub(r"\[[^\]]*\]\s*$", "", q).strip().strip('"“”')
            if len(q) >= 20:
                quotes.append(q)
            buf = []
    return quotes


_MIN_RUN = 10      # only contiguous runs this long count toward coverage
_MAX_GAP = 12      # any longer unmatched stretch inside the quote = misquote


def _segment_ok(seg, chunks, ratio=None):
    """Does this quote segment occur (near-)verbatim in any chunk?
    Coverage counts only significant contiguous runs, and any large
    unmatched stretch (e.g. inserted/altered words) fails the quote.
    Returns (ok, chunk_id)."""
    import difflib
    nseg = _norm(seg)
    if len(nseg) < 12:
        return True, None                       # too short to judge
    best = (0.0, None, len(nseg))               # coverage, id, max_gap
    for c in chunks:
        ntext = _norm(c["text"])
        if nseg in ntext:
            return True, c["id"]
        sm = difflib.SequenceMatcher(None, nseg, ntext, autojunk=False)
        mask = bytearray(len(nseg))
        for b in sm.get_matching_blocks():
            if b.size >= _MIN_RUN:
                mask[b.a:b.a + b.size] = b"\1" * b.size
        coverage = sum(mask) / len(nseg)
        gaps = [len(g) for g in bytes(mask).split(b"\1") if g]
        max_gap = max(gaps, default=0)
        if coverage > best[0]:
            best = (coverage, c["id"], max_gap)
    coverage, cid, max_gap = best
    ratio = config.QUOTE_MATCH_RATIO if ratio is None else ratio
    ok = coverage >= ratio and max_gap <= _MAX_GAP
    return ok, cid


def verify_quotes(answer, chunks, ratio=None):
    """Check every blockquote in `answer` against the retrieved chunks.
    Ellipsized quotes are verified segment by segment. `ratio` is the
    collection's QUOTE_MATCH_RATIO (default: config). Returns
    [{"text", "ok", "chunk_id"}]."""
    results = []
    for q in extract_quotes(answer):
        segs = [s for s in re.split(r"…|\.\.\.", q) if s.strip()]
        oks, cid = [], None
        for seg in segs:
            ok, chunk_id = _segment_ok(seg, chunks, ratio)
            oks.append(ok)
            cid = cid or chunk_id
        results.append({"text": q, "ok": all(oks), "chunk_id": cid})
    return results


# ------------------------------------- search only / closest material --
# P2-9: both show retrieved passages WITHOUT a model call. Neither changes
# what the gate accepts: closest_material() runs only after gate_ok() has
# said no, and search_passages() reports the gate verdict as "covered"
# without acting on it.

SNIPPET_WORDS = 40
CLOSEST_K = 3


def passage_info(c, score, snippet_words=SNIPPET_WORDS):
    """The JSON shape the UI lists: a source entry plus a text snippet."""
    words = c["text"].split()
    snippet = " ".join(words[:snippet_words])
    if len(words) > snippet_words:
        snippet += " …"
    return {"id": c["id"], "essay": c["essay"], "year": chunk_year(c),
            "page_start": c["page_start"], "page_end": c["page_end"],
            "source": c["source"],
            "work": c.get("work", Path(c["source"]).stem),
            "score": round(float(score), 4), "snippet": snippet}


def closest_material(index, query, n=CLOSEST_K):
    """For a gated-out question: the n nearest passages (one per section,
    best-first), so the user sees what the collection does hold instead
    of a bare refusal. [] when nothing matches at all."""
    out, seen = [], set()
    for s, c in index.search(query, max(n * 4, n)):
        key = (c["source"], c["essay"])
        if key in seen:
            continue
        seen.add(key)
        out.append(passage_info(c, s))
        if len(out) >= n:
            break
    return out


def search_passages(col, question, index=None, k=None):
    """Search-only mode: ranked passages for `question`, no LLM call.
    Same retrieval path as ask_stream (pool, rerank, per-source cap), no
    follow-up rewrite (that needs the model). "covered" is the gate's
    verdict, reported for the UI and never used to filter."""
    index = index or load_index(col)
    k = _cfg(col, "TOP_K") if k is None else int(k)
    k = max(1, min(k, 50))
    pool = index.search(question, k * _cfg(col, "POOL_FACTOR"))
    hits = diversify(_maybe_rerank(question, pool, _cfg(col, "RERANK")), k,
                     cap=_cfg(col, "MAX_PER_WORK"))
    return {"question": question, "covered": bool(gate_ok(index, question)),
            "results": [passage_info(c, s) for s, c in hits]}


# -------------------------------------------------------------------- ask --

def ask_stream(col, question, mode="scholar", index=None, k=None,
               history=None):
    """Returns {"sources", "mode", "stream", "answer", "dropped",
    "rewritten_question", "history_used"}, plus "closest" when gated out.

    stream is a generator of text deltas, or None when the question is
    gated out (then "answer" holds the fallback text and "closest" the
    nearest passages, closest_material()). "sources" lists only
    the excerpts actually sent, in prompt order: excerpt [n] in the prompt
    is sources[n-1]. "dropped" counts hits removed to fit the model window
    (config.LLM_CTX - config.LLM_MAX_TOKENS).

    `history`: recent turns [{question, answer}, …], oldest first (the
    caller passes at most FOLLOWUP_TURNS). When non-empty and the question
    looks like a follow-up (is_followup), ONE rewrite call produces a
    standalone question; retrieval, era parsing and the gate then use it
    ("rewritten_question", else None). The turns go into the prompt inside
    the token budget, trimmed before any excerpt ("history_used" = turns
    kept). Every tuning value is read per collection (col.setting)."""
    if mode not in available_modes(col):
        mode = "scholar"
    index = index or load_index(col)
    n_turns = max(int(_cfg(col, "FOLLOWUP_TURNS")), 0)
    history = list(history or [])[-n_turns:] if n_turns else []
    rewritten = rewrite_question(col, question, history) if history else None
    query = rewritten or question
    if not gate_ok(index, query):
        return {"answer": not_covered(col), "sources": [], "mode": mode,
                "stream": None, "dropped": 0,
                "rewritten_question": rewritten, "history_used": 0,
                "closest": closest_material(index, query)}
    system_fn, temp_key, k_key = MODES.get(mode, MODES["scholar"])
    system = system_fn(col)
    temperature = _cfg(col, temp_key or "TEMPERATURE")
    if k is None:                       # mode default unless caller set -k
        k = _cfg(col, k_key or "TOP_K")
    era = parse_era(query)
    pool = apply_era(index.search(query, k * _cfg(col, "POOL_FACTOR")),
                     era, k)
    hits = diversify(_maybe_rerank(query, pool, _cfg(col, "RERANK")), k,
                     cap=_cfg(col, "MAX_PER_WORK"))        # best-first
    model = backend = burl = None
    persona = col.persona
    if mode == "persona" and persona and persona.get("model"):
        model = persona["model"]          # fine-tuned voice model (persona only)
        backend = persona.get("backend") or None
        burl = persona.get("url") or None
        system = system_persona_tuned(col)
    # Budget against the final system prompt; trims history, then drops by
    # retrieval rank, then orders chronologically for evolution questions.
    hits, user, dropped, used = fit_prompt(
        system, question, hits, history, numbered=(mode != "quotes"),
        standalone=rewritten, order_question=query,
        answer_words=_cfg(col, "HISTORY_ANSWER_WORDS"))
    gen = llm_stream(system, user, temperature=temperature, model=model,
                     backend=backend, url=burl)
    sources = [{"id": c["id"], "essay": c["essay"], "year": chunk_year(c),
                "page_start": c["page_start"], "page_end": c["page_end"],
                "source": c["source"],
                "work": c.get("work", Path(c["source"]).stem),
                "score": round(s, 4),
                **({"truncated": True} if c.get("truncated") else {})}
               for s, c in hits]
    return {"sources": sources, "mode": mode, "stream": gen, "answer": None,
            "dropped": dropped, "rewritten_question": rewritten,
            "history_used": used}


def ask(col, question, mode="scholar", index=None, k=None, history=None):
    """Blocking variant: returns {"answer", "sources", "mode", …}; scholar
    and persona answers also get "citations" (check_citations)."""
    res = ask_stream(col, question, mode=mode, index=index, k=k,
                     history=history)
    if res["stream"] is not None:
        res["answer"] = "".join(res["stream"])
        if res["mode"] != "quotes":
            res["citations"] = check_citations(res["answer"],
                                               len(res["sources"]))
        if res["mode"] == "persona":
            res["answer"] = persona_disclaimer(col) + res["answer"]
    res.pop("stream", None)
    return res


# ----------------------------------------------------------------- runtime --

class Runtime:
    """Everything query time needs for ONE collection, loaded once: the
    retrieval index, the reader map, and the quote bank. serve.py keeps a
    small LRU of these so switching collections is instant; ask.py builds
    one. All disk access happens here at load time, so concurrent requests
    against different collections never race on shared state."""

    def __init__(self, col):
        self.col = col
        self.index = load_index(col)
        self.by_id, self.groups = build_reader(self.index.chunks)
        self.quote_bank = load_quote_bank(col)

    def ask_stream(self, question, mode="scholar", k=None, history=None):
        return ask_stream(self.col, question, mode=mode, index=self.index, k=k,
                          history=history)

    def search(self, question, k=None):
        return search_passages(self.col, question, index=self.index, k=k)

    def passage(self, chunk_id, before=1, after=1):
        return passage(self.col, self.by_id, self.groups, chunk_id,
                       before=before, after=after)

    def random_quote(self, topic=None, era=None):
        return random_quote(self.quote_bank, self.by_id, index=self.index,
                            topic=topic, era=era)
