"""M1-C: token budget (P0-6), embedded lock (P0-7), NotIndexed (P0-8)."""
import gc

import pytest

import config
import errors
import rag


def _chunk(i, words):
    return {"id": f"doc.txt-s{i}-{i * 1000}", "source": "doc.txt",
            "essay": f"Section {i}", "work": "doc", "page_start": 1,
            "page_end": 1, "text": " ".join(f"w{i}x{j}" for j in range(words))}


def _ranked(n, words):
    """Best-first hits with strictly decreasing scores."""
    return [(1.0 - i / 100, _chunk(i, words)) for i in range(n)]


def _est(system, user):
    return len((system + "\n" + user).split()) * 1.35


# ------------------------------------------------------------ token budget --

def test_dropped_hits_are_lowest_ranked(monkeypatch):
    monkeypatch.setattr(config, "LLM_CTX", 1000)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 200)
    hits = _ranked(10, 100)
    system = "You are a test system prompt."
    sent, user, dropped = rag.fit_context(system, "What is it?", hits)
    assert dropped > 0 and len(sent) + dropped == len(hits)
    assert [c["id"] for _, c in sent] == [c["id"] for _, c in hits[:len(sent)]]
    assert _est(system, user) + config.LLM_MAX_TOKENS <= config.LLM_CTX


def test_evolution_drops_by_rank_not_chronology(monkeypatch):
    monkeypatch.setattr(config, "LLM_CTX", 1000)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 200)
    hits = _ranked(10, 100)
    # Best-ranked hit is the LATEST; a chronological sort puts it last.
    for i, (_, c) in enumerate(hits):
        c["date"], c["date_precision"] = f"{2010 - i}-01-01", "year"
    sent, _, dropped = rag.fit_context("sys", "How has it changed over time?",
                                       hits)
    assert dropped > 0
    kept = {c["id"] for _, c in sent}
    assert kept == {c["id"] for _, c in hits[:len(sent)]}
    years = [rag.chunk_year(c) for _, c in sent]
    assert years == sorted(years)                 # still presented in order


def test_single_oversize_excerpt_truncated_not_omitted(monkeypatch):
    monkeypatch.setattr(config, "LLM_CTX", 400)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 100)
    hits = _ranked(3, 1000)
    original = hits[0][1]["text"]
    system = "A short system prompt."
    sent, user, dropped = rag.fit_context(system, "q?", hits)
    assert len(sent) == 1 and dropped == 2
    c = sent[0][1]
    assert c["id"] == hits[0][1]["id"] and c["truncated"]
    assert 0 < len(c["text"].split()) < 1000
    assert c["text"].split()[0] in user
    assert hits[0][1]["text"] == original         # index chunk not mutated
    assert _est(system, user) + config.LLM_MAX_TOKENS <= config.LLM_CTX


def test_everything_fits_drops_nothing(monkeypatch):
    monkeypatch.setattr(config, "LLM_CTX", 100000)
    hits = _ranked(5, 50)
    sent, _, dropped = rag.fit_context("sys", "q?", hits)
    assert dropped == 0 and len(sent) == 5
    assert not any(c.get("truncated") for _, c in sent)


def test_sources_match_what_was_sent(make_collection, fake_llm, monkeypatch):
    col = make_collection()
    chunks = rag.load_chunks(col)
    monkeypatch.setattr(config, "LLM_CTX", 1400)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 200)
    monkeypatch.setattr(rag, "gate_ok", lambda *a: True)

    class Everything:
        def __init__(self):
            self.chunks = chunks

        def search(self, q, k=10):
            return [(1.0 - i / 1000, c) for i, c in enumerate(self.chunks)][:k]

    res = rag.ask_stream(col, "Tell me everything", index=Everything(), k=10)
    "".join(res["stream"])
    prompt = fake_llm.prompt_text()
    assert res["dropped"] > 0
    assert len(res["sources"]) == prompt.count("\n\n---\n\n") + 1
    by_id = {c["id"]: c for c in chunks}
    for s in res["sources"]:
        text = by_id[s["id"]]["text"]
        if s.get("truncated"):
            text = " ".join(text.split()[:20])
        assert text in prompt


# ------------------------------------------------------------ embedded lock --

class _FakeLlama:
    def __init__(self):
        self.kw = None

    def create_chat_completion(self, **kw):
        self.kw = kw
        for t in ("a", "b", "c"):
            yield {"choices": [{"delta": {"content": t}}]}


@pytest.fixture
def embedded(monkeypatch):
    fake = _FakeLlama()
    monkeypatch.setattr(config, "LLM_BACKEND", "embedded")
    monkeypatch.setattr(rag, "_EMBEDDED_LLM", fake)
    yield fake
    assert not rag._EMBEDDED_LOCK.locked()


def test_embedded_lock_released_on_close(embedded):
    gen = rag.llm_stream("s", "u")
    assert next(gen) == "a"
    assert rag._EMBEDDED_LOCK.locked()
    gen.close()
    assert not rag._EMBEDDED_LOCK.locked()
    assert embedded.kw["max_tokens"] == config.LLM_MAX_TOKENS


def test_embedded_lock_released_on_gc(embedded):
    gen = rag.llm_stream("s", "u")
    next(gen)
    del gen
    gc.collect()
    assert not rag._EMBEDDED_LOCK.locked()


def test_embedded_lock_released_after_full_stream(embedded):
    assert "".join(rag.llm_stream("s", "u")) == "abc"
    assert not rag._EMBEDDED_LOCK.locked()


# --------------------------------------------------------------- NotIndexed --

def test_not_indexed_for_empty_collection(make_collection):
    col = make_collection(ingest=False)
    with pytest.raises(errors.NotIndexed, match="ingest.py -c"):
        rag.load_sources(col)
    assert not issubclass(errors.NotIndexed, SystemExit)


def test_not_indexed_when_chunk_files_are_empty(make_collection):
    col = make_collection(ingest=False)
    col.chunks_dir.mkdir(parents=True, exist_ok=True)
    (col.chunks_dir / "empty.jsonl").write_text("")
    with pytest.raises(errors.NotIndexed):
        rag.load_index(col)
