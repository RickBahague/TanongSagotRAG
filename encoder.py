"""The one text encoder every stage shares (corpus embedding AND queries).

get_encoder() returns a process-wide cached encoder:
  1. OnnxEncoder  when config.ONNX_ENCODER_DIR/model.onnx exists
                  (onnxruntime + tokenizers + numpy, ~60 MB — the desktop
                  build's path; no torch)
  2. STEncoder    sentence-transformers, when installed (dev)
  3. otherwise    ImportError (callers fall back to BM25)

Corpus vectors and query vectors MUST come from the same encoder with the
same pooling and max sequence length, so each encoder exposes a `tag`
({model, pooling, max_len}) that ingest writes into every vector file and
rag checks when loading them.

Pooling / max length for the ONNX path are read from the sentence-
transformers config files of config.EMBED_MODEL, stored next to model.onnx
(scripts/export_onnx.py downloads them):
  <model_dir>/1_Pooling/config.json        pooling_mode_* flags
  <model_dir>/sentence_bert_config.json    max_seq_length
If they are missing, DEFAULT_POOLING / DEFAULT_MAX_LEN are used (the values
of BAAI/bge-small-en-v1.5, the default EMBED_MODEL) and a warning is logged.
"""
import json
import logging
import threading
from pathlib import Path

import config

log = logging.getLogger("anyrag.encoder")

POOLING_FILE = "1_Pooling/config.json"
SBERT_FILE = "sentence_bert_config.json"
# Fallbacks when the config files are absent: BAAI/bge-small-en-v1.5's own
# values (CLS pooling, 512 tokens). Wrong for other models — re-run
# scripts/export_onnx.py to fetch the real files.
DEFAULT_POOLING = "cls"
DEFAULT_MAX_LEN = 512

# sentence-transformers' concatenation order when several modes are on
_MODES = (("pooling_mode_cls_token", "cls"),
          ("pooling_mode_max_tokens", "max"),
          ("pooling_mode_mean_tokens", "mean"),
          ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len"))


def read_model_config(model_dir):
    """(pooling, max_len) from the sentence-transformers config files in
    `model_dir`; documented defaults (with a warning) where absent.
    `pooling` is e.g. "cls", "mean", or "cls+mean" for concatenated modes."""
    model_dir = Path(model_dir)
    pooling, max_len = DEFAULT_POOLING, DEFAULT_MAX_LEN
    pf = model_dir / POOLING_FILE
    if pf.exists():
        cfg = json.loads(pf.read_text())
        modes = [name for key, name in _MODES if cfg.get(key)]
        if not modes:
            raise ValueError(f"{pf}: no supported pooling mode enabled")
        pooling = "+".join(modes)
    else:
        log.warning("%s missing — assuming %s pooling (run "
                    "scripts/export_onnx.py)", pf, DEFAULT_POOLING)
    sf = model_dir / SBERT_FILE
    if sf.exists():
        max_len = int(json.loads(sf.read_text()).get("max_seq_length")
                      or DEFAULT_MAX_LEN)
    else:
        log.warning("%s missing — assuming max_seq_length %d (run "
                    "scripts/export_onnx.py)", sf, DEFAULT_MAX_LEN)
    return pooling, max_len


def _batches(n, size):
    size = max(int(size or 1), 1)
    for start in range(0, n, size):
        yield start, min(start + size, n)


class OnnxEncoder:
    """config.EMBED_MODEL (bge-small-en-v1.5 by default) via onnxruntime +
    tokenizers. The exported graph returns only last_hidden_state
    (B, T, H); pooling is applied here, as configured by the model's
    sentence-transformers files, then vectors are L2-normalized."""

    def __init__(self, model_dir, model_name=None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        model_dir = Path(model_dir)
        self.pooling, self.max_len = read_model_config(model_dir)
        self.tag = {"model": model_name or config.EMBED_MODEL,
                    "pooling": self.pooling, "max_len": self.max_len}
        self._tok_file = str(model_dir / "tokenizer.json")
        self._counter = None                # untruncated twin, lazily
        self.tok = Tokenizer.from_file(self._tok_file)
        self.tok.no_padding()
        # max_len counts the special tokens, as in sentence-transformers
        self.tok.enable_truncation(max_length=self.max_len)
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.sess.get_inputs()}
        outs = {o.name: o for o in self.sess.get_outputs()}
        self._out = ("last_hidden_state" if "last_hidden_state" in outs
                     else next(iter(outs)))
        h = outs[self._out].shape[-1]
        n_modes = len(self.pooling.split("+"))
        self.dim = h * n_modes if isinstance(h, int) else None

    def token_count(self, text):
        """Tokens (incl. special tokens) BEFORE truncation — compare with
        max_len to see how much of a text is not embedded."""
        if self._counter is None:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(self._tok_file)
            t.no_truncation()
            t.no_padding()
            self._counter = t
        return len(self._counter.encode(text).ids)

    def _pool(self, hidden, mask):
        np = self.np
        m = mask[..., None].astype(hidden.dtype)
        parts = []
        for mode in self.pooling.split("+"):
            if mode == "cls":
                parts.append(hidden[:, 0])
            elif mode == "max":
                parts.append(np.where(m > 0, hidden, -1e9).max(1))
            elif mode == "mean":
                parts.append((hidden * m).sum(1) / m.sum(1).clip(min=1e-9))
            elif mode == "mean_sqrt_len":
                parts.append((hidden * m).sum(1)
                             / np.sqrt(m.sum(1).clip(min=1e-9)))
            else:
                raise ValueError(f"unsupported pooling mode {mode!r}")
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _encode_batch(self, texts):
        np = self.np
        encs = self.tok.encode_batch(list(texts))
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype="int64")
        mask = np.zeros_like(ids)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = e.attention_mask
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run([self._out], feeds)[0]
        if hidden.ndim == 2:                # graph already pooled
            return hidden
        return self._pool(hidden, mask)

    def encode(self, texts, batch_size=None, normalize_embeddings=True,
               progress=None, **_):
        """(N, H) float32. Texts are length-sorted into batches of
        `batch_size` (config.EMBED_BATCH) to keep padding small; output is
        in input order. progress(done, total) is called after each batch."""
        np = self.np
        texts = list(texts)
        n = len(texts)
        if not n:
            return np.zeros((0, self.dim or 0), dtype="float32")
        order = sorted(range(n), key=lambda i: len(texts[i]))
        out = [None] * n
        for a, b in _batches(n, batch_size or config.EMBED_BATCH):
            idx = order[a:b]
            vecs = self._encode_batch([texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
            if progress:
                progress(b, n)
        vecs = np.vstack(out).astype("float32")
        self.dim = vecs.shape[1]
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs


class STEncoder:
    """sentence-transformers fallback (dev installs). Same interface and
    tag as OnnxEncoder; pooling/max_len are read from the loaded model."""

    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or config.EMBED_MODEL)
        self.max_len = int(self.model.max_seq_length)
        pooling = DEFAULT_POOLING
        try:
            from sentence_transformers.models import Pooling
            for mod in self.model:
                if isinstance(mod, Pooling):
                    cfg = mod.get_config_dict()
                    pooling = "+".join(name for key, name in _MODES
                                       if cfg.get(key)) or pooling
        except Exception:
            log.warning("could not read pooling from %s — tagging %s",
                        model_name or config.EMBED_MODEL, DEFAULT_POOLING)
        self.pooling = pooling
        self.tag = {"model": model_name or config.EMBED_MODEL,
                    "pooling": self.pooling, "max_len": self.max_len}
        self.dim = None

    def token_count(self, text):
        return len(self.model.tokenizer(text)["input_ids"])

    def encode(self, texts, batch_size=None, normalize_embeddings=True,
               progress=None, **_):
        import numpy as np
        texts = list(texts)
        n = len(texts)
        if not n:
            return np.zeros((0, self.dim or 0), dtype="float32")
        bs = batch_size or config.EMBED_BATCH
        parts = []
        for a, b in _batches(n, bs):
            parts.append(self.model.encode(
                texts[a:b], batch_size=bs, show_progress_bar=False,
                normalize_embeddings=normalize_embeddings))
            if progress:
                progress(b, n)
        vecs = np.vstack(parts).astype("float32")
        self.dim = vecs.shape[1]
        return vecs


_ENCODER = None
_LOCK = threading.Lock()


def get_encoder():
    """The process-wide encoder (created once, thread-safe). Raises
    ImportError when neither the ONNX encoder nor sentence-transformers is
    available. A failed attempt is not cached, so installing the encoder
    later works without a restart."""
    global _ENCODER
    if _ENCODER is None:
        with _LOCK:
            if _ENCODER is None:
                if (Path(config.ONNX_ENCODER_DIR) / "model.onnx").exists():
                    _ENCODER = OnnxEncoder(config.ONNX_ENCODER_DIR)
                else:
                    _ENCODER = STEncoder()      # ImportError if absent
    return _ENCODER


def current_tag():
    """The {model, pooling, max_len} tag vectors must carry to be usable
    with the current encoder, or None when no encoder is available. On the
    ONNX path this reads only the config files (no model load)."""
    if _ENCODER is not None:
        return dict(_ENCODER.tag)
    if (Path(config.ONNX_ENCODER_DIR) / "model.onnx").exists():
        pooling, max_len = read_model_config(config.ONNX_ENCODER_DIR)
        return {"model": config.EMBED_MODEL, "pooling": pooling,
                "max_len": max_len}
    try:
        return dict(get_encoder().tag)
    except ImportError:
        return None


def tag_mismatch(npz, tag=None):
    """Why vectors in the loaded .npz `npz` can't be used with the current
    encoder (a message), or None when they can. Missing pooling/max_len
    tags (vectors written before they existed) count as a mismatch."""
    tag = tag if tag is not None else current_tag()
    files = set(npz.files)
    model = str(npz["model"]) if "model" in files else None
    if model != config.EMBED_MODEL:
        return f"embedded with {model} but EMBED_MODEL is {config.EMBED_MODEL}"
    if tag is None:
        return None          # no encoder to compare against; model matches
    for key in ("pooling", "max_len"):
        have = str(npz[key]) if key in files else None
        if have != str(tag[key]):
            return f"{key} is {have} but the encoder uses {tag[key]}"
    return None


class OnnxReranker:
    """Cross-encoder (config.RERANK_MODEL, ms-marco-MiniLM-L-6-v2 by
    default) via onnxruntime + tokenizers: scores (query, passage) pairs.
    Same .predict(pairs) interface as sentence-transformers' CrossEncoder;
    returns the raw logit per pair (higher = more relevant)."""

    def __init__(self, model_dir, max_len=None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len or config.RERANK_MAX_LEN,
                                   strategy="only_second")
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.sess.get_inputs()}

    def predict(self, pairs, batch_size=16):
        np = self.np
        pairs = list(pairs)
        out = []
        for a, b in _batches(len(pairs), batch_size):
            encs = self.tok.encode_batch([tuple(p) for p in pairs[a:b]])
            maxlen = max(len(x.ids) for x in encs)
            ids = np.zeros((len(encs), maxlen), dtype="int64")
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for i, x in enumerate(encs):
                ids[i, :len(x.ids)] = x.ids
                mask[i, :len(x.ids)] = x.attention_mask
                types[i, :len(x.ids)] = x.type_ids
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = types
            logits = self.sess.run(None, feeds)[0]
            out.extend(float(v) for v in logits.reshape(len(encs), -1)[:, 0])
        return out


def get_reranker():
    """OnnxReranker when config.ONNX_RERANKER_DIR holds the model, else
    sentence-transformers' CrossEncoder; ImportError when neither."""
    d = Path(config.ONNX_RERANKER_DIR)
    if (d / "model.onnx").exists() and (d / "tokenizer.json").exists():
        return OnnxReranker(d)
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL)
