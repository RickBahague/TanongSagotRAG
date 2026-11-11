"""RAG core: retrieval over chunked sources + generation via a configurable
local LLM backend (Ollama or any OpenAI-compatible server).

Retrieval (config.RETRIEVAL):
  bm25    pure-Python BM25 (stdlib only) — always available
  dense   encoder.py vectors (data/vectors/; ONNX or sentence-transformers)
  hybrid  BM25 + dense fused with reciprocal-rank fusion (RRF)
  auto    hybrid when vectors + libs exist, else bm25

Generation supports streaming (llm_stream / ask_stream) and blocking (ask).
"""
import json
import math
import re
import threading
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


def tokenize(text):
    return [_light_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in STOPWORDS and len(w) > 1]


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

class BM25Index:
    """Section titles are indexed WITH the chunk text — segmentation strips
    headings from the body, so 'deep sea mining' must still match a chunk
    whose section is titled that even when the body words differ."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c.get("essay", "") + ". " + c["text"])
                     for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=config.TOP_K):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                num = tf[t] * (config.BM25_K1 + 1)
                den = tf[t] + config.BM25_K1 * (1 - config.BM25_B
                        + config.BM25_B * self.doc_len[i] / self.avg_len)
                s += self.idf.get(t, 0) * num / den
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scores[:k]]


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

    def search(self, query, k=config.TOP_K):
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

    def __init__(self, sources):
        self.chunks = [c for _, chunks, _ in sources for c in chunks]
        self.bm25 = BM25Index(self.chunks)
        self.dense = DenseIndex(sources)

    def search(self, query, k=config.TOP_K):
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

    def __init__(self, summaries):
        self.entries = [e for _, es, _ in summaries for e in es]
        self._texts = [f"{e['essay']}. {e['summary']} {e['keywords']}"
                       for e in self.entries]
        self.bm25 = BM25Index([{"text": t, **e} for t, e in
                               zip(self._texts, self.entries)])
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

    def search(self, query, k=config.TOP_K):
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
    if mode == "bm25" or (mode == "auto" and not have_vectors):
        base = BM25Index(chunks)
    else:
        cls = {"dense": DenseIndex}.get(mode, HybridIndex)  # auto/hybrid → hybrid
        try:
            base = cls(sources)
        except ImportError:
            print("[rag] vectors found but no query encoder available "
                  "— using BM25. (bash setup.sh, or bundle the ONNX encoder)")
            base = BM25Index(chunks)
        except Exception as e:
            print(f"[rag] {cls.__name__} unavailable ({e}) — using BM25. "
                  f"Re-run: .venv/bin/python ingest.py -c {col.id} --force")
            base = BM25Index(chunks)

    if config.HIERARCHICAL in ("auto", "on"):
        summaries = load_summaries(col)
        if summaries:
            try:
                return HierarchicalIndex(base, SummaryIndex(summaries))
            except Exception as e:
                print(f"[rag] hierarchical retrieval unavailable ({e}) — flat")
        elif config.HIERARCHICAL == "on":
            print("[rag] HIERARCHICAL=on but no summaries — run summarize.py")
    return base


_RERANKER = None


def _maybe_rerank(query, hits):
    """Optional cross-encoder reordering of the candidate pool (S4b)."""
    global _RERANKER
    if config.RERANK == "off" or not hits:
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

def _gate_min_score(n_docs):
    """MIN_SCORE was tuned on a ~5k-chunk corpus; BM25 scores scale roughly
    with log(corpus size) via the idf term, so scale the gate down for small
    collections instead of blocking everything they could answer."""
    return config.MIN_SCORE * min(1.0, math.log1p(n_docs) / math.log1p(5000))


def bm25_gate_ok(bm, question, with_hits=False):
    """BM25 signal: top score >= the log-scaled MIN_SCORE and a top-k hit
    covering >= MIN_COVERAGE of the query terms."""
    hits = bm.search(question, config.TOP_K)
    qtoks = set(tokenize(question))
    cov = max((sum(1 for t in qtoks
                   if t in set(tokenize(c.get("essay", "") + " " + c["text"])))
               / len(qtoks) for _, c in hits), default=0) if qtoks else 0
    ok = bool(hits and hits[0][0] >= _gate_min_score(len(bm.chunks))
              and cov >= config.MIN_COVERAGE)
    return (ok, hits) if with_hits else ok


def dense_top_sim(dn, question):
    """Raw cosine of the best-matching chunk (0.0 when nothing is > 0)."""
    hits = dn.search(question, 1)
    return hits[0][0] if hits else 0.0


def gate_ok(index, question):
    """Is the question answerable from the corpus?

    BM25-only index: the BM25 signal alone (unchanged behaviour).
    With a dense signal (cosine of the top chunk), bge puts unrelated text
    at ~0.3-0.64, so dense can't pass by a low bar and BM25 can't pass on
    its own either (it false-accepts on shared words):
        cos >= DENSE_SOLO_SIM                  — dense alone is convincing
        or (BM25 passes and cos >= MIN_DENSE_SIM) — both signals agree
    Thresholds and the measurements behind them: config.py."""
    dn = index if isinstance(index, DenseIndex) else getattr(index, "dense", None)
    bm = index if isinstance(index, BM25Index) else getattr(index, "bm25", None)
    if dn is None:
        return bool(bm) and bm25_gate_ok(bm, question)
    if bm is None:                  # dense-only retrieval: still gate on both
        bm = getattr(dn, "_gate_bm25", None)
        if bm is None:
            bm = dn._gate_bm25 = BM25Index(dn.chunks)
    cos = dense_top_sim(dn, question)
    if cos >= config.DENSE_SOLO_SIM:
        return True
    ok, hits = bm25_gate_ok(bm, question, with_hits=True)
    if not ok:
        return False
    if cos >= config.MIN_DENSE_SIM:
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


def _embedded_llm():
    """Lazy-load the bundled GGUF once (first question pays the load cost)."""
    global _EMBEDDED_LLM
    if _EMBEDDED_LLM is None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "LLM_BACKEND is 'embedded' but llama-cpp-python is not "
                "installed (pip install llama-cpp-python).")
        if not Path(config.LLM_MODEL_PATH).exists():
            raise RuntimeError(f"Model file missing: {config.LLM_MODEL_PATH}")
        kwargs = {"model_path": str(config.LLM_MODEL_PATH),
                  "n_ctx": config.LLM_CTX, "verbose": False}
        if config.LLM_THREADS:
            kwargs["n_threads"] = config.LLM_THREADS
        _EMBEDDED_LLM = Llama(**kwargs)
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
        raise RuntimeError(f"{config.LLM_BACKEND} backend error for model "
                           f"'{model}': {detail}") from None


def llm_stream(system, user, temperature=None, model=None, backend=None,
               url=None):
    """Yield text deltas from the configured backend (config.LLM_BACKEND).
    `model`/`backend`/`url` override the defaults for this call (used by
    persona mode's dedicated voice model/server)."""
    temperature = config.TEMPERATURE if temperature is None else temperature
    model = model or config.LLM_MODEL
    backend = backend or config.LLM_BACKEND
    url = url or config.LLM_URL
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if backend == "ollama":
        req = _request(f"{url}/api/chat", {
            "model": model, "messages": messages, "stream": True,
            "options": {"temperature": temperature,
                        "num_ctx": config.LLM_CTX,
                        "num_predict": config.LLM_MAX_TOKENS}})
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
            "max_tokens": config.LLM_MAX_TOKENS})
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
                max_tokens=config.LLM_MAX_TOKENS)
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


def system_scholar(col):
    return f"""You are a careful research assistant answering questions about \
{_collection_label(col)}. Answer ONLY from the excerpts provided. Quote short \
passages where useful. Cite every claim as [Section Title, p. N] using the metadata \
given with each excerpt. Excerpt headers may include the year the piece was written; \
when the question concerns how views changed or developed over time, organize the \
answer chronologically and name the years. If the excerpts do not answer the \
question, say the documents provided do not cover it. Never invent facts."""


def system_persona(col):
    return f"""You speak in the first person as {col.author}, in the direct \
style of the source texts. Ground every statement strictly in the excerpts \
provided from their writings; do not invent positions. If the excerpts do not \
cover the question, say plainly that your writings here do not address it. Keep \
the voice consistent with the excerpts."""


# When a fine-tuned voice model is active the style lives in the weights;
# only the grounding contract remains in the prompt.
def system_persona_tuned(col):
    return f"""You are {col.author}. Ground every statement strictly in the \
excerpts provided from your writings; do not invent positions. If the excerpts \
do not cover the question, say plainly that your writings here do not address it."""


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


def build_context(hits):
    parts = []
    for _, c in hits:
        pages = (f"p. {c['page_start']}" if c["page_start"] == c["page_end"]
                 else f"pp. {c['page_start']}-{c['page_end']}")
        work = c.get("work", Path(c["source"]).stem)
        label = c["essay"] if c["essay"] == work else f"{c['essay']} — {work}"
        y = chunk_year(c)
        year = f", {y}" if y else ""
        parts.append(f"[{label}{year}, {pages}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------ token budget --
# Estimated, not tokenized: words x 1.35 over every message sent. Integer
# form (words x 135 vs. tokens x 100) avoids float edge cases at the limit.

_TOK_PER_WORD_X100 = 135


def _words(text):
    return len(text.split())


def user_prompt(context, question):
    return (f"Excerpts from the documents:\n\n{context}"
            f"\n\nQuestion: {question}")


def _order_for_prompt(hits, question):
    """Evolution questions present excerpts chronologically; otherwise the
    retrieval rank order is kept."""
    if _EVOLUTION.search(question):
        return sorted(hits, key=lambda h: (chunk_year(h[1]) is None,
                                           chunk_year(h[1]) or 0))
    return list(hits)


def fit_context(system, question, hits):
    """Budget the prompt to the model window.

    `hits` must be ranked best-first. Returns (sent_hits, user, dropped):
    the lowest-ranked hits are removed until
    words(system + user) x 1.35 + LLM_MAX_TOKENS <= LLM_CTX. If even the
    best hit alone does not fit, its text is truncated (on a copy; the
    indexed chunk is never mutated, and the copy carries "truncated": True)
    rather than sending no excerpt. sent_hits are in presentation order
    (chronological for evolution questions). Config is read at call time."""
    budget_x100 = (config.LLM_CTX - config.LLM_MAX_TOKENS) * 100
    sys_words = _words(system)

    def build(kept):
        ordered = _order_for_prompt(kept, question)
        user = user_prompt(build_context(ordered), question)
        ok = (sys_words + _words(user)) * _TOK_PER_WORD_X100 <= budget_x100
        return ok, ordered, user

    kept = list(hits)
    while kept:
        ok, ordered, user = build(kept)
        if ok:
            return ordered, user, len(hits) - len(kept)
        if len(kept) == 1:
            break
        kept.pop()                    # lowest retrieval rank goes first
    if not kept:
        return [], user_prompt("", question), 0

    # A single excerpt is still over budget: truncate its text to fit.
    s, c = kept[0]
    text_words = c["text"].split()
    _, _, user = build(kept)
    overhead = sys_words + _words(user) - len(text_words)
    allowed = max(budget_x100 // _TOK_PER_WORD_X100 - overhead - 1, 0)  # "…"
    short = dict(c, text=" ".join(text_words[:allowed] + ["…"]),
                 truncated=True)
    _, ordered, user = build([(s, short)])
    return ordered, user, len(hits) - 1


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
    """Word offset encoded as the last id segment (see ingest.py).
    Deduped ids carry a '#n' suffix (see dedup_ids) — strip it."""
    return int(c["id"].rsplit("-", 1)[-1].split("#")[0])


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
    (CHUNK_OVERLAP_WORDS) removed at every stitch so nothing repeats.
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
                if _chunk_start(ch) - _chunk_start(prev) == step:
                    skip = ov
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

    drop = 0
    if prevs:
        if _chunk_start(c) - _chunk_start(prevs[-1]) == step:
            drop = ov      # before-block tail duplicates the cited chunk head
    before_text = assemble(prevs, drop_tail=drop)
    if prevs and not drop:
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


def _segment_ok(seg, chunks):
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
    ok = coverage >= config.QUOTE_MATCH_RATIO and max_gap <= _MAX_GAP
    return ok, cid


def verify_quotes(answer, chunks):
    """Check every blockquote in `answer` against the retrieved chunks.
    Ellipsized quotes are verified segment by segment. Returns
    [{"text", "ok", "chunk_id"}]."""
    results = []
    for q in extract_quotes(answer):
        segs = [s for s in re.split(r"…|\.\.\.", q) if s.strip()]
        oks, cid = [], None
        for seg in segs:
            ok, chunk_id = _segment_ok(seg, chunks)
            oks.append(ok)
            cid = cid or chunk_id
        results.append({"text": q, "ok": all(oks), "chunk_id": cid})
    return results


# -------------------------------------------------------------------- ask --

def ask_stream(col, question, mode="scholar", index=None, k=None):
    """Returns {"sources", "mode", "stream", "answer", "dropped"}.
    stream is a generator of text deltas, or None when the question is
    gated out (then "answer" holds the fallback text). "sources" lists only
    the excerpts actually sent; "dropped" counts hits removed to fit the
    model window (config.LLM_CTX - config.LLM_MAX_TOKENS)."""
    if mode not in available_modes(col):
        mode = "scholar"
    index = index or load_index(col)
    if not gate_ok(index, question):
        return {"answer": not_covered(col), "sources": [], "mode": mode,
                "stream": None, "dropped": 0}
    system_fn, temp_key, k_key = MODES.get(mode, MODES["scholar"])
    system = system_fn(col)
    temperature = getattr(config, temp_key) if temp_key else None
    if k is None:                       # mode default unless caller set -k
        k = getattr(config, k_key) if k_key else config.TOP_K
    era = parse_era(question)
    pool = apply_era(index.search(question, k * config.POOL_FACTOR), era, k)
    hits = diversify(_maybe_rerank(question, pool), k)   # best-first
    model = backend = burl = None
    persona = col.persona
    if mode == "persona" and persona and persona.get("model"):
        model = persona["model"]          # fine-tuned voice model (persona only)
        backend = persona.get("backend") or None
        burl = persona.get("url") or None
        system = system_persona_tuned(col)
    # Budget against the final system prompt; drops by retrieval rank, then
    # orders chronologically for evolution questions (see fit_context).
    hits, user, dropped = fit_context(system, question, hits)
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
            "dropped": dropped}


def ask(col, question, mode="scholar", index=None, k=None):
    """Blocking variant: returns {"answer", "sources", "mode"}."""
    res = ask_stream(col, question, mode=mode, index=index, k=k)
    if res["stream"] is not None:
        res["answer"] = "".join(res["stream"])
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

    def ask_stream(self, question, mode="scholar", k=None):
        return ask_stream(self.col, question, mode=mode, index=self.index, k=k)

    def passage(self, chunk_id, before=1, after=1):
        return passage(self.col, self.by_id, self.groups, chunk_id,
                       before=before, after=after)

    def random_quote(self, topic=None, era=None):
        return random_quote(self.quote_bank, self.by_id, index=self.index,
                            topic=topic, era=era)
