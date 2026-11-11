"""M2-A: one shared encoder (P0-1), pooling/max_len from the model's
sentence-transformers config (P0-2), tagged vectors, batched embed with
progress, import_work through embed_source."""
import importlib.util
import json
import threading

import pytest

import config
import encoder
import ingest
import library
import rag
from helpers import CORPUS

HAVE_MODEL = ((config.ONNX_ENCODER_DIR / "model.onnx").exists()
              and all(importlib.util.find_spec(m)
                      for m in ("onnxruntime", "tokenizers", "numpy")))
needs_model = pytest.mark.skipif(
    not HAVE_MODEL, reason="ONNX encoder not present: models/encoder-onnx/"
    "model.onnx missing (run scripts/export_onnx.py) or onnxruntime/"
    "tokenizers/numpy not installed")


def _npz(path):
    import numpy as np
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _rewrite(path, **changes):
    """Rewrite a vector file with some tags changed (None = drop the tag)."""
    import numpy as np
    d = _npz(path)
    for k, v in changes.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = np.array(v)
    np.savez(path, **d)


# ------------------------------------------------------------ config read --

def test_read_model_config_modes(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps(
        {"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True}))
    (tmp_path / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 256}))
    assert encoder.read_model_config(tmp_path) == ("mean", 256)


def test_read_model_config_defaults_warn(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="anyrag.encoder"):
        got = encoder.read_model_config(tmp_path)
    assert got == (encoder.DEFAULT_POOLING, encoder.DEFAULT_MAX_LEN)
    assert "1_Pooling" in caplog.text and "sentence_bert_config" in caplog.text


@needs_model
def test_shipped_config_files_present():
    d = config.ONNX_ENCODER_DIR
    assert (d / encoder.POOLING_FILE).exists(), "run scripts/export_onnx.py"
    assert (d / encoder.SBERT_FILE).exists(), "run scripts/export_onnx.py"
    cfg = json.loads((d / encoder.POOLING_FILE).read_text())
    want = "+".join(n for k, n in encoder._MODES if cfg.get(k))
    enc = encoder.get_encoder()
    assert enc.pooling == want
    assert enc.max_len == json.loads(
        (d / encoder.SBERT_FILE).read_text())["max_seq_length"]


# --------------------------------------------------------------- encoder --

@needs_model
def test_get_encoder_is_one_cached_instance():
    got = []
    ts = [threading.Thread(target=lambda: got.append(encoder.get_encoder()))
          for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len({id(e) for e in got}) == 1
    assert got[0] is encoder.get_encoder() is rag._query_encoder()
    assert isinstance(got[0], rag.OnnxEncoder)      # compat alias


@needs_model
def test_onnx_pooling_matches_config():
    """The vector is the configured pooling of last_hidden_state,
    L2-normalized (CLS for bge-small-en-v1.5)."""
    import numpy as np
    enc = encoder.get_encoder()
    text = "Worker bees secrete wax to build the comb."
    e = enc.tok.encode(text)
    ids = np.array([e.ids], dtype="int64")
    mask = np.array([e.attention_mask], dtype="int64")
    feeds = {"input_ids": ids, "attention_mask": mask,
             "token_type_ids": np.zeros_like(ids)}
    feeds = {k: v for k, v in feeds.items() if k in enc._inputs}
    hidden = enc.sess.run(["last_hidden_state"], feeds)[0]
    want = enc._pool(hidden, mask)[0]
    if enc.pooling == "cls":
        assert np.allclose(want, hidden[0, 0])
    want = want / np.linalg.norm(want)
    got = enc.encode([text])[0]
    assert np.allclose(got, want, atol=1e-5)
    assert abs(float(np.linalg.norm(got)) - 1) < 1e-5


@needs_model
def test_encode_batches_and_order(monkeypatch):
    import numpy as np
    enc = encoder.get_encoder()
    texts = ["short", "a much longer sentence about volcanoes and lava",
             "mid length text here", "x"]
    calls = []
    monkeypatch.setattr(config, "EMBED_BATCH", 2)
    batched = enc.encode(texts, progress=lambda d, n: calls.append((d, n)))
    assert calls == [(2, 4), (4, 4)]
    single = np.vstack([enc.encode([t])[0] for t in texts])
    assert np.allclose(batched, single, atol=1e-5)
    assert enc.encode([]).shape[0] == 0


@needs_model
def test_truncates_at_max_len():
    enc = encoder.get_encoder()
    long = "lava " * (enc.max_len * 2)
    assert enc.token_count(long) > enc.max_len
    assert len(enc.tok.encode(long).ids) == enc.max_len


def test_no_encoder_raises_import_error(monkeypatch, tmp_path):
    if importlib.util.find_spec("sentence_transformers"):
        pytest.skip("sentence-transformers installed: it is the fallback")
    monkeypatch.setattr(encoder, "_ENCODER", None)
    monkeypatch.setattr(config, "ONNX_ENCODER_DIR", tmp_path)
    with pytest.raises(ImportError):
        encoder.get_encoder()
    assert encoder.current_tag() is None


def test_no_encoder_embed_skips_and_bm25(monkeypatch, tmp_path,
                                         make_collection):
    if importlib.util.find_spec("sentence_transformers"):
        pytest.skip("sentence-transformers installed: it is the fallback")
    monkeypatch.setattr(encoder, "_ENCODER", None)
    monkeypatch.setattr(config, "ONNX_ENCODER_DIR", tmp_path)
    col = make_collection(files=("volcanoes.txt",))
    st = [e["status"] for e in col.ingest_events if e["stage"] == "embed"]
    assert st and st[-1].startswith("skipped")
    assert not list(col.vectors_dir.glob("*.npz"))
    assert isinstance(rag.load_index(library.get(col.id)), rag.BM25Index)


# ------------------------------------------------------------ embed stage --

@needs_model
def test_vectors_tagged(make_collection):
    import numpy as np
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    enc = encoder.get_encoder()
    for name in ("volcanoes.txt", "beekeeping.md"):
        key = col.artifact_key(name)
        d = _npz(col.vectors_dir / (key + ".npz"))
        assert {"vectors", "model", "pooling", "max_len", "src_hash"} <= set(d)
        assert str(d["model"]) == config.EMBED_MODEL
        assert str(d["pooling"]) == enc.pooling
        assert int(d["max_len"]) == enc.max_len
        cf = col.chunks_dir / (key + ".jsonl")
        assert str(d["src_hash"]) == ingest.file_hash(cf)
        n = sum(1 for l in open(cf) if l.strip())
        assert d["vectors"].shape == (n, enc.dim)
        assert np.allclose(np.linalg.norm(d["vectors"], axis=1), 1, atol=1e-5)
    assert not list(col.vectors_dir.glob("*.tmp*"))


@needs_model
def test_embed_progress_events(make_collection, monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH", 2)
    col = make_collection(files=("beekeeping.md",))
    n = ingest.load_manifest(col)["beekeeping.md"]["n_chunks"]
    prog = [e for e in col.ingest_events
            if e["stage"] == "embed" and "done" in e]
    assert prog and all(e["source"] == "beekeeping.md" for e in prog)
    assert [e["done"] for e in prog] == sorted(e["done"] for e in prog)
    assert prog[-1]["done"] == prog[-1]["total"] == n
    assert len(prog) == -(-n // 2)                  # one per batch
    final = [e for e in col.ingest_events
             if e["stage"] == "embed" and "done" not in e]
    assert final[-1]["status"].startswith("embedded")


@needs_model
def test_vectors_stale_on_pooling_change(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    key = col.artifact_key("volcanoes.txt")
    out = col.vectors_dir / (key + ".npz")
    cf = col.chunks_dir / (key + ".jsonl")
    n = ingest.load_manifest(col)["volcanoes.txt"]["n_chunks"]
    assert not ingest.vectors_stale(out, cf, n)

    other = "mean" if encoder.get_encoder().pooling != "mean" else "cls"
    _rewrite(out, pooling=other)
    assert ingest.vectors_stale(out, cf, n)
    _rewrite(out, pooling=encoder.get_encoder().pooling, max_len=128)
    assert ingest.vectors_stale(out, cf, n)
    _rewrite(out, pooling=None, max_len=None)      # pre-M2 file: no tags
    assert ingest.vectors_stale(out, cf, n)
    # explicit tag argument is honoured too
    tag = dict(encoder.current_tag(), pooling=other)
    assert ingest.vectors_stale(out, cf, n, tag=tag)

    # a plain re-ingest re-embeds it with the current tags
    from helpers import reingest
    events = reingest(col)
    st = [e["status"] for e in events
          if e["stage"] == "embed" and "done" not in e]
    assert st == ["embedded %d chunks" % n]
    assert str(_npz(out)["pooling"]) == encoder.get_encoder().pooling
    assert not ingest.vectors_stale(out, cf, n)
    st = [e["status"] for e in reingest(col) if e["stage"] == "embed"]
    assert st == ["up to date"]


# ------------------------------------------------------------- retrieval --

def _base(index):
    return index.base if isinstance(index, rag.HierarchicalIndex) else index


@needs_model
def test_hybrid_index_loads(make_collection):
    col = make_collection()                         # whole fixture corpus
    index = rag.load_index(library.get(col.id))
    base = _base(index)
    assert isinstance(base, rag.HybridIndex)
    assert isinstance(index.dense, rag.DenseIndex)
    assert len(index.dense.chunks) == len(index.chunks)
    hits = index.dense.search("How do worker bees build the wax comb?", 3)
    assert hits and hits[0][1]["source"] == "beekeeping.md"


@needs_model
def test_dense_index_falls_back_on_pooling_mismatch(make_collection, capsys):
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    _rewrite(col.vectors_dir / (col.artifact_key("volcanoes.txt") + ".npz"),
             pooling="mean" if encoder.get_encoder().pooling != "mean"
             else "cls")
    index = rag.load_index(library.get(col.id))
    assert isinstance(_base(index), rag.BM25Index)
    assert "pooling" in capsys.readouterr().out
    with pytest.raises(ValueError, match="pooling"):
        rag.DenseIndex(rag.load_sources(library.get(col.id)))


@needs_model
def test_summary_index_ignores_mismatched_vectors(make_collection):
    import numpy as np
    col = make_collection(files=("volcanoes.txt",))
    key = col.artifact_key("volcanoes.txt")
    entries = [{"essay": "Lava", "source": "volcanoes.txt",
                "summary": "about lava", "keywords": "lava magma"}]
    vec = col.vectors_dir / (key + ".summaries.npz")
    enc = encoder.get_encoder()
    good = dict(vectors=enc.encode(["Lava. about lava lava magma"]),
                model=np.array(config.EMBED_MODEL),
                pooling=np.array(enc.pooling), max_len=np.array(enc.max_len),
                src_hash=np.array("x"))
    np.savez(vec, **good)
    si = rag.SummaryIndex([(key, entries, vec)])
    assert si.vecs is not None and si.model is enc
    np.savez(vec, **dict(good, max_len=np.array(enc.max_len // 2)))
    si = rag.SummaryIndex([(key, entries, vec)])
    assert si.vecs is None and si.model is None     # BM25 explore only


@needs_model
def test_query_and_embed_share_encoder(make_collection):
    """Same encoder object and identical vectors for the same text, whether
    it was embedded by the ingest stage or encoded as a query would be."""
    import numpy as np
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    index = rag.load_index(library.get(col.id))
    assert index.dense.model is encoder.get_encoder()
    key = col.artifact_key("beekeeping.md")
    chunks = [json.loads(l) for l in open(col.chunks_dir / (key + ".jsonl"))]
    stored = _npz(col.vectors_dir / (key + ".npz"))["vectors"]
    for i in (0, len(chunks) - 1):
        q = index.dense.model.encode([chunks[i]["text"]],
                                     normalize_embeddings=True)[0]
        assert np.allclose(q, stored[i], atol=1e-5)


# ---------------------------------------------------------------- server --

@needs_model
def test_import_work_embeds(make_collection):
    import serve
    col = make_collection(files=(), linked=False, ingest=False)
    data = (CORPUS / "volcanoes.txt").read_bytes()
    res = serve.import_work(col, "volcanoes.txt", data)
    assert res["embedded"] is True
    assert res["retrieval"] in ("HybridIndex", "HierarchicalIndex")
    key = col.artifact_key("volcanoes.txt")
    d = _npz(col.vectors_dir / (key + ".npz"))
    assert str(d["pooling"]) == encoder.get_encoder().pooling
    assert int(d["max_len"]) == encoder.get_encoder().max_len
    cf = col.chunks_dir / (key + ".jsonl")
    assert not ingest.vectors_stale(col.vectors_dir / (key + ".npz"), cf,
                                    res["chunks"])
    m = ingest.load_manifest(col)["volcanoes.txt"]
    assert m["embed_status"].startswith("embedded") and m["embedded_at"]
    # the next plain ingest finds it up to date (same format as the pipeline)
    from helpers import reingest
    st = [e["status"] for e in reingest(col) if e["stage"] == "embed"]
    assert st == ["up to date"]
