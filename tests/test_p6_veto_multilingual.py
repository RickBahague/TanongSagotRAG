"""Reranker veto on the gate (English-only) and per-model encoder profiles."""
import json

import pytest

import config
import encoder
import rag


# ------------------------------------------------------------- language --

@pytest.mark.parametrize("q,want", [
    ("How do I join a political party in Canada?", True),
    ("Recipe for chocolate cake", True),
    ("Why is summer hot in Marseille?", True),
    ("STRASSENBAHN ZÜRICH", False),                       # keyword query
    ("Quelle est la recette du gâteau au chocolat ?", False),
    ("Τι είναι η κβαντική φυσική;", False),
    ("Wie entstehen Vulkane?", False),
    ("", False),
])
def test_english_question_detection(q, want):
    assert rag._is_english_question(q) is want


# ------------------------------------------------------------------ veto --

class Scored:
    """Fake cross-encoder: score by a keyword in the passage."""
    def __init__(self, good="comb"):
        self.good, self.calls = good, 0

    def predict(self, pairs):
        self.calls += 1
        return [3.0 if self.good in p[1].lower() else -8.0 for p in pairs]


@pytest.fixture
def veto_on(monkeypatch):
    monkeypatch.setattr(config, "RERANK", "auto")
    monkeypatch.setattr(config, "RERANK_GATE", "auto")


def test_veto_rejects_when_no_passage_answers(make_collection, veto_on, monkeypatch):
    index = rag.load_index(make_collection())
    q = "How do worker bees build the wax comb?"
    assert rag._signal_gate_ok(index, q)
    monkeypatch.setattr(rag, "_RERANKER", Scored(good="zzz-nothing"))
    assert not rag.gate_ok(index, q)
    rag._RERANK_MEMO.clear()
    monkeypatch.setattr(rag, "_RERANKER", Scored(good="comb"))
    assert rag.gate_ok(index, q)


def test_veto_skipped_when_off_or_not_english(make_collection, veto_on, monkeypatch):
    index = rag.load_index(make_collection())
    q = "How do worker bees build the wax comb?"
    fake = Scored(good="zzz-nothing")
    monkeypatch.setattr(rag, "_RERANKER", fake)
    monkeypatch.setattr(config, "RERANK_GATE", "off")
    assert rag.gate_ok(index, q) and fake.calls == 0
    monkeypatch.setattr(config, "RERANK_GATE", "auto")
    monkeypatch.setattr(config, "RERANK", "off")
    assert rag.gate_ok(index, q) and fake.calls == 0
    monkeypatch.setattr(config, "RERANK", "auto")
    monkeypatch.setattr(rag, "_RERANKER", False)          # unavailable
    assert rag.gate_ok(index, q)


def test_veto_never_accepts_what_the_signals_reject(make_collection, veto_on, monkeypatch):
    index = rag.load_index(make_collection())
    monkeypatch.setattr(rag, "_RERANKER", Scored(good=""))     # scores all 3.0
    assert not rag.gate_ok(index, "Who won the 2018 FIFA World Cup?")


def test_rerank_order_skipped_for_non_english(monkeypatch):
    fake = Scored()
    monkeypatch.setattr(rag, "_RERANKER", fake)
    hits = [(1.0, {"id": "a", "text": "x"}), (0.5, {"id": "b", "text": "comb"})]
    assert rag._maybe_rerank("Πώς χτίζουν οι μέλισσες την κηρήθρα;", hits, "auto") is hits
    assert fake.calls == 0
    out = rag._maybe_rerank("How do bees build the comb?", hits, "auto")
    assert [c["id"] for _, c in out] == ["b", "a"]


def test_rerank_scores_memoized(monkeypatch):
    fake = Scored()
    monkeypatch.setattr(rag, "_RERANKER", fake)
    hits = [(1.0, {"id": "a", "text": "comb"}), (0.5, {"id": "b", "text": "x"})]
    rag._maybe_rerank("How do bees build the comb?", hits, "auto")
    rag._maybe_rerank("How do bees build the comb?", hits, "auto")
    assert fake.calls == 1


def test_veto_settings_in_schema():
    for k in ("RERANK_GATE", "RERANK_MIN_SCORE"):
        assert k in config.SETTINGS_SCHEMA
    assert config.check_value("RERANK_MIN_SCORE", -1.35) is None
    assert config.check_value("RERANK_GATE", "sometimes") is not None


# -------------------------------------------------------- model profiles --

def test_profiles():
    bge = config.encoder_profile("BAAI/bge-small-en-v1.5")
    assert bge["dir"] == "encoder-onnx" and bge["passage_prefix"] == ""
    e5 = config.encoder_profile("intfloat/multilingual-e5-small")
    assert (e5["query_prefix"], e5["passage_prefix"]) == ("query: ", "passage: ")
    assert e5["gate"] == (0.75, 0.77)
    other = config.encoder_profile("someone/my-e5-large")
    assert other["dir"] == "encoder-onnx-my-e5-large"
    assert other["query_prefix"] == "query: " and "gate" not in other


def test_switching_model_applies_and_restores_profile(monkeypatch):
    f = config.SETTINGS_FILE
    before = (config.MIN_DENSE_SIM, config.DENSE_SOLO_SIM, config.ONNX_ENCODER_DIR,
              config.EMBED_QUERY_PREFIX)
    try:
        f.write_text(json.dumps({"EMBED_MODEL": "intfloat/multilingual-e5-small"}))
        config.reload_settings()
        assert (config.MIN_DENSE_SIM, config.DENSE_SOLO_SIM) == (0.75, 0.77)
        assert config.ONNX_ENCODER_DIR.name == "encoder-onnx-multilingual-e5-small"
        assert config.EMBED_PASSAGE_PREFIX == "passage: "
        # a user value beats the profile
        f.write_text(json.dumps({"EMBED_MODEL": "intfloat/multilingual-e5-small",
                                 "MIN_DENSE_SIM": 0.8}))
        config.reload_settings()
        assert (config.MIN_DENSE_SIM, config.DENSE_SOLO_SIM) == (0.8, 0.77)
    finally:
        f.unlink(missing_ok=True)
        config.reload_settings()
    assert (config.MIN_DENSE_SIM, config.DENSE_SOLO_SIM, config.ONNX_ENCODER_DIR,
            config.EMBED_QUERY_PREFIX) == before
    assert config.EMBED_PASSAGE_PREFIX == ""


def test_encoder_reloads_when_model_changes(monkeypatch):
    class Fake:
        tag = {"model": "old/model"}
    monkeypatch.setattr(encoder, "_ENCODER", Fake())
    made = []
    monkeypatch.setattr(encoder, "STEncoder", lambda: made.append(1) or Fake())
    monkeypatch.setattr(config, "ONNX_ENCODER_DIR", config.ROOT / "no-such-dir")
    encoder.get_encoder()
    assert made == [1]


def test_passage_prefix_applied_at_embed(make_collection, monkeypatch):
    seen = []
    real = encoder.get_encoder

    class Spy:
        def __init__(self, enc):
            self.enc, self.tag = enc, enc.tag

        def encode(self, texts, **kw):
            seen.extend(texts)
            return self.enc.encode(texts, **kw)
    try:
        real()
    except ImportError:
        pytest.skip("no encoder")
    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "passage: ")
    monkeypatch.setattr(encoder, "get_encoder", lambda: Spy(real()))
    make_collection(files=("volcanoes.txt",))
    assert seen and all(t.startswith("passage: ") for t in seen)
