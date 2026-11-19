"""M5-2 (P2-9): "closest material" for gated-out questions and a search-only
mode. Neither calls the model and neither changes gate decisions.

(The idf-weighted coverage part of P2-9 was measured and reverted — see
plan.md "M5-2 eval" — so the gate here is the unchanged M5-1 gate.)"""
import shutil

import pytest

import library
import rag
from helpers import CLEAN_FILES, CORPUS, gate_expected, load_eval

# off-topic, but shares one word ("volcanoes") with the fixtures, so
# BM25-only installs have closest material to show as well (coverage 1/6)
OFF = "How do I configure a Kubernetes ingress controller near volcanoes?"
ON = "How do worker bees build the wax comb?"
KEYS = {"id", "essay", "year", "page_start", "page_end", "source", "work",
        "score", "snippet"}


@pytest.fixture
def no_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(rag, "llm_stream", boom)
    monkeypatch.setattr(rag, "llm_chat", boom)


@pytest.fixture
def col(make_collection):
    return library.get(make_collection("M52").id)


def test_gated_out_returns_closest_material(col, no_llm):
    index = rag.load_index(col)
    assert not rag.gate_ok(index, OFF)
    res = rag.ask_stream(col, OFF, index=index)
    assert res["stream"] is None and res["sources"] == []
    assert res["answer"] == rag.not_covered(col)
    closest = res["closest"]
    assert 1 <= len(closest) <= rag.CLOSEST_K
    assert all(set(c) == KEYS for c in closest)
    ids = {c["id"] for c in index.chunks}
    assert all(c["id"] in ids for c in closest)
    # one entry per section, best-first
    assert len({(c["source"], c["essay"]) for c in closest}) == len(closest)
    scores = [c["score"] for c in closest]
    assert scores == sorted(scores, reverse=True)


def test_answered_question_has_no_closest_key(col, fake_llm):
    res = rag.ask_stream(col, ON)
    assert res["stream"] is not None and "closest" not in res
    "".join(res["stream"])


def test_closest_material_empty_when_nothing_matches(col):
    index = rag.load_index(col)
    bm = index.bm25 if hasattr(index, "bm25") else index
    assert rag.closest_material(bm, "zzqx qqzv") == []


def test_snippet_is_clipped_text():
    c = {"id": "x#0", "essay": "E", "page_start": 1, "page_end": 1,
         "source": "x.md", "text": " ".join(f"w{i}" for i in range(100))}
    p = rag.passage_info(c, 1.23456)
    assert p["snippet"].split()[:rag.SNIPPET_WORDS] == \
        [f"w{i}" for i in range(rag.SNIPPET_WORDS)]
    assert p["snippet"].endswith(" …") and p["score"] == 1.2346
    c["text"] = "short text"
    assert rag.passage_info(c, 0)["snippet"] == "short text"


def test_search_only_no_model_call(col, no_llm):
    index = rag.load_index(col)
    res = rag.search_passages(col, ON, index=index)
    assert res["question"] == ON and res["covered"] is True
    assert res["results"] and res["results"][0]["essay"] == "Hive Structure"
    assert all(set(r) == KEYS for r in res["results"])
    assert len(res["results"]) <= col.setting("TOP_K")
    off = rag.search_passages(col, OFF, index=index, k=3)
    assert off["covered"] is False and len(off["results"]) <= 3
    assert len(rag.search_passages(col, ON, index=index, k=500)["results"]) \
        <= 50


def test_gate_decisions_unchanged(col):
    """Search-only / closest material must not alter what the gate does."""
    index = rag.load_index(col)
    for q in load_eval():
        before = rag.gate_ok(index, q["q"])
        rag.search_passages(col, q["q"], index=index)
        rag.closest_material(index, q["q"])
        assert rag.gate_ok(index, q["q"]) == before == gate_expected(index, q)


# ------------------------------------------------------------------ HTTP --

def _indexed(server, tmp_path):
    d = tmp_path / "m52-docs"
    d.mkdir()
    for f in CLEAN_FILES:
        shutil.copy2(CORPUS / f, d / f)
    st, c = server.json("POST", "/collections", {"name": "M52 web",
                                                  "link": str(d)})
    assert st == 200
    st, ev = server.ndjson("POST", f"/c/{c['id']}/ingest?summaries=0")
    assert ev[-1]["stage"] == "complete"
    return c["id"]


def test_http_search_and_closest(server, fake_llm, tmp_path):
    cid = _indexed(server, tmp_path)
    n_calls = len(fake_llm.requests)
    st, res = server.json("POST", f"/c/{cid}/search", {"question": ON})
    assert st == 200 and res["covered"] is True
    assert res["results"][0]["essay"] == "Hive Structure"
    st, res = server.json("POST", f"/c/{cid}/search",
                          {"question": OFF, "k": 2})
    assert st == 200 and res["covered"] is False and len(res["results"]) <= 2
    # gated-out /ask: meta carries the closest material, no model call
    st, evs = server.ndjson("POST", f"/c/{cid}/ask", {"question": OFF})
    assert st == 200 and evs[0]["type"] == "meta"
    assert evs[0]["sources"] == [] and evs[0]["closest"]
    assert set(evs[0]["closest"][0]) == KEYS
    assert len(fake_llm.requests) == n_calls
    st, evs = server.ndjson("POST", f"/c/{cid}/ask", {"question": ON})
    assert evs[0]["closest"] == [] and evs[0]["sources"]


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": 3},
                                  {"question": "x", "k": 0},
                                  {"question": "x", "k": "5"},
                                  {"question": "x", "k": True}])
def test_http_search_validation(server, tmp_path, body):
    cid = _indexed(server, tmp_path)
    st, res = server.json("POST", f"/c/{cid}/search", body)
    assert st == 400 and "error" in res


def test_http_search_not_indexed(server, tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    st, c = server.json("POST", "/collections", {"name": "Empty",
                                                  "link": str(d)})
    st, res = server.json("POST", f"/c/{c['id']}/search", {"question": ON})
    assert st == 409
    st, res = server.json("POST", "/c/nope/search", {"question": ON})
    assert st == 404
