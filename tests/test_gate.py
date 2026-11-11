"""Relevance gate + retrieval quality on the fixture corpus (BM25 path).

These pin today's behaviour, including one known false-reject, so a
retrieval change shows up as a test diff rather than going unnoticed.
Update the expectations deliberately (and record eval numbers in plan.md).
"""
import re

import pytest

import rag
from helpers import load_eval

QUESTIONS = load_eval()
ON = [q for q in QUESTIONS if q["expect"]]
OFF = [q for q in QUESTIONS if not q["expect"]]
# known BM25 false-reject: 2 of 5 query terms in the chunk (< MIN_COVERAGE)
KNOWN_FALSE_REJECTS = {"What can ice cores tell us about past climate?"}


@pytest.fixture
def index(make_collection):
    return rag.load_index(make_collection())


def test_eval_file_shape():
    assert len(ON) == 15 and len(OFF) == 5


@pytest.mark.parametrize("q", ON, ids=lambda q: q["q"][:40])
def test_expected_section_in_top3(index, q):
    rx = re.compile(q["expect"], re.I)
    top = index.search(q["q"], 3)
    assert any(rx.search(c["essay"]) or rx.search(c.get("work", ""))
               or rx.search(c["text"]) for _, c in top)


@pytest.mark.parametrize("q", ON, ids=lambda q: q["q"][:40])
def test_gate_accepts_on_topic(index, q):
    assert rag.gate_ok(index, q["q"]) is (q["q"] not in KNOWN_FALSE_REJECTS)


@pytest.mark.parametrize("q", OFF, ids=lambda q: q["q"][:40])
def test_gate_rejects_off_topic(index, q):
    assert not rag.gate_ok(index, q["q"])


def test_gated_question_skips_llm(make_collection, fake_llm):
    col = make_collection()
    res = rag.ask_stream(col, "Recipe for chocolate cake")
    assert res["stream"] is None and res["sources"] == []
    assert "do not appear to cover" in res["answer"]
    assert fake_llm.requests == []


def test_diversify_caps_per_source():
    hits = [(10 - i, {"source": "a" if i < 8 else "b", "id": str(i)})
            for i in range(10)]
    out = rag.diversify(hits, 8, cap=3)
    srcs = [c["source"] for _, c in out]
    # both b's make it in although they rank last; a backfills the rest
    assert len(out) == 8 and srcs.count("b") == 2 and srcs[:3] == ["a"] * 3
