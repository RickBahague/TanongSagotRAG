#!/usr/bin/env python3
"""Fetch the ONNX query encoder for desktop builds.

The config.EMBED_MODEL repo on Hugging Face (default BAAI/bge-small-en-v1.5)
ships a pre-exported onnx/model.onnx, so no torch/optimum export is needed —
just download it:

  .venv/bin/python scripts/export_onnx.py

Produces models/encoder-onnx/{model.onnx,tokenizer.json} (the directory
config.ONNX_ENCODER_DIR points at). Once present, rag.OnnxEncoder is used
automatically for query encoding. If sentence-transformers is installed,
output is sanity-checked against it.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

OUT = config.ONNX_ENCODER_DIR
BASE = f"https://huggingface.co/{config.EMBED_MODEL}/resolve/main"
FILES = {"model.onnx": f"{BASE}/onnx/model.onnx",
         "tokenizer.json": f"{BASE}/tokenizer.json"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT / name
        if dest.exists():
            print(f"already present: {dest.name}")
            continue
        print(f"downloading {name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")

    # verify the encoder loads and, if possible, matches sentence-transformers
    from rag import OnnxEncoder
    texts = ["land reform and national democracy"]
    a = OnnxEncoder(OUT).encode(texts)[0]
    print(f"OnnxEncoder OK (dim={len(a)})")
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        b = SentenceTransformer(config.EMBED_MODEL).encode(
            texts, normalize_embeddings=True)[0]
        cos = float(np.dot(a, b))
        print(f"cosine vs sentence-transformers: {cos:.4f}")
        assert cos > 0.999, "ONNX output mismatch!"
    except ImportError:
        print("(sentence-transformers not installed — skipped cross-check)")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
