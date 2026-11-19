#!/usr/bin/env python3
"""Fetch the ONNX query encoder for desktop builds.

The config.EMBED_MODEL repo on Hugging Face (default BAAI/bge-small-en-v1.5)
ships a pre-exported onnx/model.onnx, so no torch/optimum export is needed —
just download it:

  .venv/bin/python scripts/export_onnx.py
  .venv/bin/python scripts/export_onnx.py --model intfloat/multilingual-e5-small

Produces, in models/encoder-onnx/ (config.ONNX_ENCODER_DIR):
  model.onnx, tokenizer.json
  1_Pooling/config.json, sentence_bert_config.json   (sentence-transformers
      pooling mode and max_seq_length — encoder.OnnxEncoder reads these so
      its vectors match sentence-transformers')
Files already present are kept; missing ones are downloaded, so re-running
this on an older models/ dir adds just the two config files. Once present,
encoder.get_encoder() uses the ONNX encoder for both corpus embedding and
queries. If sentence-transformers is installed, output is sanity-checked
against it.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

if "--model" in sys.argv:            # fetch another model's folder
    config.EMBED_MODEL = sys.argv[sys.argv.index("--model") + 1]
OUT = config.ROOT / "models" / config.encoder_profile()["dir"]
BASE = f"https://huggingface.co/{config.EMBED_MODEL}/resolve/main"
FILES = {"model.onnx": f"{BASE}/onnx/model.onnx",
         "tokenizer.json": f"{BASE}/tokenizer.json",
         "1_Pooling/config.json": f"{BASE}/1_Pooling/config.json",
         "sentence_bert_config.json": f"{BASE}/sentence_bert_config.json"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT / name
        if dest.exists():
            print(f"already present: {name}")
            continue
        print(f"downloading {name} …")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")

    # verify the encoder loads and, if possible, matches sentence-transformers
    from encoder import OnnxEncoder
    texts = ["How do glaciers carve valleys over thousands of years?"]
    enc = OnnxEncoder(OUT, config.EMBED_MODEL)
    a = enc.encode(texts)[0]
    print(f"OnnxEncoder OK (dim={len(a)}, pooling={enc.pooling}, "
          f"max_len={enc.max_len})")
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
