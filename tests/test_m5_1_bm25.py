"""M5-1 (P2-5): inverted-index BM25, gate reusing hits, on-disk token cache.

The pre-P2-5 full-scan scorer is kept below as the reference: rankings and
scores must be identical, not merely close.
"""
import json
import math
import random
from collections import Counter

import pytest

import config
import rag
from helpers import load_eval


class OldBM25:
    """Verbatim scoring of BM25Index before P2-5 (full scan per query)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [rag.tokenize(c.get("essay", "") + ". " + c["text"])
                     for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def search(self, query, k):
        q = rag.tokenize(query)
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


def _same(a, b):
    assert [(s, c["id"]) for s, c in a] == [(s, c["id"]) for s, c in b]


def _fixture_chunks(col):
    return [c for _, cs, _ in rag.load_sources(col) for c in cs]


def test_rankings_identical_on_fixtures(make_collection):
    chunks = _fixture_chunks(make_collection())
    old, new = OldBM25(chunks), rag.BM25Index(chunks)
    for q in load_eval():
        for k in (1, 3, 10, None):
            _same(new.search(q["q"], k if k else len(chunks)),
                  old.search(q["q"], k if k else len(chunks)))


def test_rankings_identical_on_synthetic_corpus():
    rnd = random.Random(7)
    vocab = [f"w{i}" for i in range(400)]
    weights = [1 / (i + 1) for i in range(len(vocab))]      # Zipfian
    chunks = [{"id": f"c{i}", "essay": rnd.choice(["Alpha", "Beta", ""]),
               "text": " ".join(rnd.choices(vocab, weights,
                                            k=rnd.randint(5, 80)))}
              for i in range(600)]
    # duplicated texts force exact ties: order must still match
    chunks += [dict(c, id=c["id"] + "dup") for c in chunks[:50]]
    old, new = OldBM25(chunks), rag.BM25Index(chunks)
    for _ in range(150):
        q = " ".join(rnd.choices(vocab, weights, k=rnd.randint(1, 6)))
        for k in (1, 8, 40, len(chunks)):
            _same(new.search(q, k), old.search(q, k))
    assert new.search("zzz-not-there", 5) == []


def test_gate_decisions_identical(make_collection, monkeypatch):
    col = make_collection()
    index = rag.load_index(col)
    chunks = index.chunks
    expected = {}
    for q in load_eval():
        expected[q["q"]] = rag.gate_ok(index, q["q"])
    # same decisions with a fresh index and with the caller's hits reused
    index2 = rag.load_index(col)
    bm = getattr(index2, "bm25", index2)
    for q, want in expected.items():
        hits = bm.search(q, None if not chunks else len(chunks))
        assert rag.gate_ok(index2, q, bm25_hits=hits) is want
        assert rag.gate_ok(index2, q) is want


def test_gate_reuses_passed_hits(make_collection, monkeypatch):
    index = rag.load_index(make_collection())
    bm = getattr(index, "bm25", index)
    q = "How do worker bees build the wax comb?"
    hits = bm.search(q, 10)
    calls = []
    orig = rag.BM25Index.search
    monkeypatch.setattr(rag.BM25Index, "search",
                        lambda self, *a, **k: calls.append(a) or
                        orig(self, *a, **k))
    rag.bm25_gate_ok(bm, q, hits=hits)
    assert calls == []
    rag.bm25_gate_ok(bm, q)
    assert len(calls) == 1


def test_repeat_query_scored_once(make_collection, monkeypatch):
    chunks = _fixture_chunks(make_collection())
    bm = rag.BM25Index(chunks)
    n = []
    orig = rag.BM25Index._term_weights
    monkeypatch.setattr(rag.BM25Index, "_term_weights",
                        lambda self, *a: n.append(1) or orig(self, *a))
    q = "glacier ice cores climate"
    a = bm.search(q, 5)
    first = len(n)
    assert first > 0
    assert bm.search(q, 3) == a[:3] and len(n) == first


def test_k1_change_invalidates_memo(make_collection, monkeypatch):
    chunks = _fixture_chunks(make_collection())
    bm = rag.BM25Index(chunks)
    q = "Fresnel lens lighthouse"
    bm.search(q, 3)
    monkeypatch.setattr(config, "BM25_K1", 2.0)
    _same(bm.search(q, 3), OldBM25(chunks).search(q, 3))


def _cache_files(col):
    d = rag._token_cache_dir(col)
    return sorted(d.glob("*.json")) if d.is_dir() else []


def test_token_cache_written_and_used(make_collection, monkeypatch):
    col = make_collection()
    rag.load_index(col)
    files = _cache_files(col)
    assert files and all(f.parent.is_relative_to(col.data_dir)
                         for f in files)
    calls = []
    orig = rag.tokenize
    monkeypatch.setattr(rag, "tokenize",
                        lambda s: calls.append(len(s)) or orig(s))
    sources = rag.load_sources(col)
    toks = rag.cached_text_tokens(col, sources)
    assert calls == []                       # every chunk text from cache
    assert toks == [orig(c["text"]) for _, cs, _ in sources for c in cs]


def test_token_cache_invalidated_on_content_change(make_collection):
    col = make_collection()
    sources = rag.load_sources(col)
    rag.cached_text_tokens(col, sources)
    key, chunks, vecs = sources[0]
    changed = [dict(chunks[0], text="entirely new zebra words")] + chunks[1:]
    toks = rag.cached_text_tokens(col, [(key, changed, vecs)] + sources[1:])
    assert toks[0] == rag.tokenize("entirely new zebra words")
    d = json.loads((rag._token_cache_dir(col) / (key + ".json")).read_text())
    assert d["hash"] == rag._texts_hash(changed)


@pytest.mark.parametrize("junk", ["", "{not json", '{"tag": "v0"}',
                                  '{"tag": null, "hash": 1, "lens": 3}'])
def test_corrupt_token_cache_tolerated(make_collection, junk):
    col = make_collection()
    rag.load_index(col)
    for f in _cache_files(col):
        f.write_text(junk)
    index = rag.load_index(col)                # rebuilds silently
    q = load_eval()[0]["q"]
    bm = getattr(index, "bm25", index)
    _same(bm.search(q, 5), OldBM25(index.chunks).search(q, 5))
    assert all(json.loads(f.read_text())["tag"] == rag._tokenizer_tag()
               for f in _cache_files(col))


def test_token_cache_drops_files_of_removed_sources(make_collection):
    col = make_collection()
    sources = rag.load_sources(col)
    rag.cached_text_tokens(col, sources)
    n = len(_cache_files(col))
    rag.cached_text_tokens(col, sources[1:])
    assert len(_cache_files(col)) == n - 1


def test_title_tokens_not_cached(make_collection):
    # a renamed section (source-meta edit) must be searchable at once
    col = make_collection()
    sources = rag.load_sources(col)
    toks = rag.cached_text_tokens(col, sources)
    chunks = [dict(c, essay="Quixotic Heading")
              for _, cs, _ in sources for c in cs]
    bm = rag.BM25Index(chunks, toks)
    assert bm.search("quixotic", 1)


def test_nonpositive_k_matches_old_slice(make_collection):
    chunks = _fixture_chunks(make_collection())
    old, new = OldBM25(chunks), rag.BM25Index(chunks)
    q = load_eval()[0]["q"]
    for k in (0, -1, -3):
        _same(new.search(q, k), old.search(q, k))


def test_empty_stem_token_round_trips_cache(make_collection):
    # "'s" stems to "" — the cache's space-joined form must still round-trip
    col = make_collection()
    sources = rag.load_sources(col)
    key, chunks, vecs = sources[0]
    odd = [dict(chunks[0], text="'s alpha 's"), dict(chunks[0], text="'s")]
    src = [(key, odd, vecs)]
    first = rag.cached_text_tokens(col, src)
    assert rag.cached_text_tokens(col, src) == first == \
        [rag.tokenize(c["text"]) for c in odd]


# ---- M5-1 re-check (M5 agent): the gate's own BM25 in dense-only mode
def test_dense_only_gate_bm25_uses_collection_stopwords(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    (col.root / "stopwords.txt").write_text("magma\n")
    import library
    col = library.get(col.id)

    class FakeDense(rag.DenseIndex):          # no encoder needed
        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, query, k=None):
            return [(0.6, self.chunks[0])]    # between the two dense bars
    dn = rag._attach_col(FakeDense(rag.load_chunks(col)), col)
    rag.gate_ok(dn, "magma")
    assert dn._gate_bm25.stopwords == rag.collection_stopwords(col)
    assert "magma" in dn._gate_bm25.stopwords


def test_collection_stopwords_without_collection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stopwords.txt").write_text("volcano\n")   # must be ignored
    assert rag.collection_stopwords(None) == frozenset(rag.STOPWORDS)


def test_ascii_fast_path_matches_unicode_path():
    """tokenize()'s ASCII fast path == the general Unicode path."""
    rnd = random.Random(7)
    alphabet = "abcXYZ019'_ -.,!?\t\n\"()#@&é"[:-1]     # ASCII only
    texts = ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 80)))
             for _ in range(3000)]
    texts += [c["q"] for c in load_eval()]
    for t in texts:
        assert t.isascii()
        slow = [rag._light_stem(w) for w in
                rag._TOKEN_RX.findall(rag.normalize_text(t))
                if w not in rag.STOPWORDS and len(w) > 1]
        assert rag.tokenize(t) == slow, repr(t)
