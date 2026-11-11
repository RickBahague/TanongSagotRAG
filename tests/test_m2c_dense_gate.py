"""M2-C: the dense gate uses raw cosine with its own thresholds
(config.MIN_DENSE_SIM / DENSE_SOLO_SIM) instead of cos*20 >= MIN_SCORE,
which passed every question once bge vectors existed."""
import importlib.util
import json

import pytest

import config
import library
import rag
from helpers import load_eval

HAVE_MODEL = ((config.ONNX_ENCODER_DIR / "model.onnx").exists()
              and all(importlib.util.find_spec(m)
                      for m in ("onnxruntime", "tokenizers", "numpy")))
needs_model = pytest.mark.skipif(
    not HAVE_MODEL, reason="ONNX encoder not present: models/encoder-onnx/"
    "model.onnx missing (run scripts/export_onnx.py) or onnxruntime/"
    "tokenizers/numpy not installed")

QUESTIONS = load_eval()
ON = [q["q"] for q in QUESTIONS if q["expect"]]
OFF = [q["q"] for q in QUESTIONS if not q["expect"]]
BEES = [q["q"] for q in QUESTIONS
        if q["expect"] in ("Hive Structure", "Honey Harvest", "Swarming")]
NOT_BEES = [q for q in ON if q not in BEES]


def _dense(index):
    return index if isinstance(index, rag.DenseIndex) else index.dense


# ------------------------------------------------------------ config ----

def test_dense_thresholds_are_raw_cosine():
    assert 0 < config.MIN_DENSE_SIM < config.DENSE_SOLO_SIM < 1


def test_dense_thresholds_overridable(monkeypatch, tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"MIN_DENSE_SIM": 0.5, "DENSE_SOLO_SIM": 0.8}))
    monkeypatch.setattr(config, "SETTINGS_FILE", f)
    monkeypatch.setattr(config, "MIN_DENSE_SIM", config.MIN_DENSE_SIM)
    monkeypatch.setattr(config, "DENSE_SOLO_SIM", config.DENSE_SOLO_SIM)
    config._apply_settings()
    assert config.MIN_DENSE_SIM == 0.5 and config.DENSE_SOLO_SIM == 0.8


# ------------------------------------------------- model-dependent ----

@needs_model
def test_dense_search_returns_raw_cosine(make_collection):
    index = rag.load_index(make_collection())
    hits = _dense(index).search(ON[0], 5)
    assert hits and all(0 < s <= 1.0 + 1e-5 for s, _ in hits)


@needs_model
def test_dense_gate_rejects_off_topic(make_collection):
    index = rag.load_index(make_collection())
    assert _dense(index) is not None                 # really gating on dense
    for q in OFF:
        assert not rag.gate_ok(index, q), q
    for q in ON:                    # incl. the old BM25 false-reject
        assert rag.gate_ok(index, q), q


@needs_model
def test_dense_gate_cross_topic_single_collection(make_collection):
    """Realistic negatives: a beekeeping-only collection asked about the
    other fixture topics (top-1 cosines up to ~0.50 here)."""
    index = rag.load_index(make_collection("Bees", files=("beekeeping.md",)))
    assert _dense(index) is not None
    for q in BEES:
        assert rag.gate_ok(index, q), q
    for q in NOT_BEES + OFF:
        assert not rag.gate_ok(index, q), q


@needs_model
def test_dense_gate_vetoes_bm25_word_overlap(make_collection):
    """BM25 alone accepts 'pyroclastic flows' on a glaciers-only collection
    (shared word 'flows'); the dense floor rejects it (cos ~0.53)."""
    col = make_collection("Ice", files=("glaciers.docx",))
    q = "What are pyroclastic flows?"
    index = rag.load_index(col)
    assert rag.bm25_gate_ok(index.bm25, q)
    assert not rag.gate_ok(index, q)


@needs_model
def test_dense_only_retrieval_still_gates(make_collection):
    col = make_collection("Dense", settings={"RETRIEVAL": "dense"})
    index = rag.load_index(library.get(col.id))
    assert isinstance(index, rag.DenseIndex)
    assert rag.gate_ok(index, ON[0])
    assert not any(rag.gate_ok(index, q) for q in OFF)


# ------------------------------------------------------ BM25-only path ----

def _old_bm25_gate(bm, question):
    """The pre-M2-C BM25 branch of gate_ok, verbatim."""
    hits = bm.search(question, config.TOP_K)
    qtoks = set(rag.tokenize(question))
    cov = max((sum(1 for t in qtoks
                   if t in set(rag.tokenize(c.get("essay", "") + " " + c["text"])))
               / len(qtoks) for _, c in hits), default=0) if qtoks else 0
    return bool(hits and hits[0][0] >= rag._gate_min_score(len(bm.chunks))
                and cov >= config.MIN_COVERAGE)


def test_bm25_only_gate_unchanged(make_collection):
    col = make_collection("Plain", settings={"RETRIEVAL": "bm25"})
    index = rag.load_index(library.get(col.id))
    assert isinstance(index, rag.BM25Index)
    got = {q["q"]: rag.gate_ok(index, q["q"]) for q in QUESTIONS}
    assert got == {q: _old_bm25_gate(index, q) for q in got}
    # pinned M0 behaviour: one known false-reject, no false-accepts
    assert [q for q in ON if not got[q]] == \
        ["What can ice cores tell us about past climate?"]
    assert not any(got[q] for q in OFF)


def test_bm25_only_gate_ignores_dense_thresholds(make_collection, monkeypatch):
    index = rag.BM25Index(rag.load_chunks(make_collection(
        "Plain2", settings={"RETRIEVAL": "bm25"})))
    before = [rag.gate_ok(index, q["q"]) for q in QUESTIONS]
    monkeypatch.setattr(config, "MIN_DENSE_SIM", 0.99)
    monkeypatch.setattr(config, "DENSE_SOLO_SIM", 0.0)
    assert [rag.gate_ok(index, q["q"]) for q in QUESTIONS] == before
