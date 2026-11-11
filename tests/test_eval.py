"""eval.py (P1-1): metrics, gate scoring, --json, --variants, CLI errors.

Runs on the fixture corpus. The bm25 variant explicitly uses rag.BM25Index,
so its numbers hold whether or not ingest also produced vectors."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import eval as ev
import rag
from helpers import FIXTURES, REPO, load_eval

ICE = "What can ice cores tell us about past climate?"


@pytest.fixture
def col(make_collection):
    c = make_collection("Eval")
    shutil.copy2(FIXTURES / "eval.jsonl", c.data_dir / "eval.jsonl")
    return c


def _cli(*args):
    env = dict(os.environ)          # ANYRAG_HOME already points at the temp home
    return subprocess.run([sys.executable, str(REPO / "eval.py"), *args],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=300)


def _consistent(m, k=10):
    for key in ("hit@1", "hit@3", f"hit@{k}", f"recall@{k}", "mrr"):
        assert 0.0 <= m[key] <= 1.0, key
    assert m["hit@1"] <= m["hit@3"] <= m[f"hit@{k}"]
    assert m["mrr"] <= m[f"hit@{k}"]
    assert m["hit@1"] <= m["mrr"]


def test_bm25_variant_known_numbers(col):
    qs = load_eval()
    chunks = rag.load_chunks(col)
    m = ev.evaluate(rag.BM25Index(chunks), qs, chunks, 10)
    _consistent(m)
    assert m["retrieval"] == "BM25Index"
    assert m["questions_on_topic"] == sum(1 for q in qs if q["expect"])
    assert m["questions_off_topic"] == sum(1 for q in qs if not q["expect"])
    assert m["hit@10"] == 1.0
    assert m["misses"] == []
    assert m["gate_false_accepts"] == 0 and m["false_accepts"] == []
    assert m["gate_false_rejects"] == 1 and m["false_rejects"] == [ICE]


def test_null_expect_not_counted_as_miss(col):
    qs = [{"q": "How do worker bees build the wax comb?", "expect": "Hive Structure"},
          {"q": "What is the capital of Mongolia?", "expect": None}]
    chunks = rag.load_chunks(col)
    m = ev.evaluate(rag.BM25Index(chunks), qs, chunks, 10)
    assert m["questions_on_topic"] == 1 and m["questions_off_topic"] == 1
    assert m["hit@10"] == 1.0 and m["misses"] == []


def test_recall_definition(col):
    chunks = rag.load_chunks(col)
    q = [{"q": "How do worker bees build the wax comb?", "expect": "Hive Structure"}]
    m1 = ev.evaluate(rag.BM25Index(chunks), q, chunks, 1)
    relevant = sum(1 for c in chunks if ev.matches(re.compile("Hive Structure", re.I), c))
    assert relevant >= 1
    assert m1["recall@1"] == pytest.approx(min(1, relevant) / relevant)
    # a regex nothing in the collection matches scores 0, not a crash
    m0 = ev.evaluate(rag.BM25Index(chunks),
                     [{"q": "bees", "expect": "zzz-no-such-section"}], chunks, 10)
    assert m0["recall@10"] == 0.0 and m0["mrr"] == 0.0


def test_variants_bm25_and_unavailable_marked(col):
    out = ev.run_variants(col, load_eval(), 10)
    assert set(out) == {"bm25", "hybrid", "hierarchical", "+rerank"}
    bm = out["bm25"]
    assert bm["retrieval"] == "BM25Index"
    _consistent(bm)
    assert bm["hit@10"] == 1.0 and bm["false_accepts"] == []
    assert bm["false_rejects"] == [ICE]
    # summaries=False in the fixture ingest
    assert out["hierarchical"].startswith("unavailable:")
    for v in out.values():
        assert isinstance(v, dict) or v.startswith("unavailable: ")
        if isinstance(v, dict):
            _consistent(v)


def test_cli_json(col):
    r = _cli("-c", col.id, "--json")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["collection"] == col.id and doc["k"] == 10
    assert doc["n_questions"] == len(load_eval())
    assert isinstance(doc["retrieval"], str)
    for key in ("gate_false_accepts", "gate_false_rejects",
                "false_accepts", "false_rejects"):
        assert key in doc
    _consistent(doc)


def test_cli_json_variants(col):
    r = _cli("-c", col.id, "--json", "--variants", "-k", "5")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["k"] == 5
    bm = doc["variants"]["bm25"]
    _consistent(bm, k=5)
    assert bm["false_rejects"] == [ICE]


def test_cli_table(col):
    r = _cli("-c", col.id)
    assert r.returncode == 0, r.stderr
    assert "hit@1" in r.stdout and "current" in r.stdout
    assert len(r.stdout.splitlines()) < 15


def test_missing_eval_file_exits(make_collection):
    c = make_collection("NoEval")
    r = _cli("-c", c.id)
    assert r.returncode != 0
    assert "No eval questions" in r.stderr
    with pytest.raises(SystemExit) as e:
        ev.main(["-c", c.id])
    assert e.value.code and "No eval questions" in str(e.value.code)
