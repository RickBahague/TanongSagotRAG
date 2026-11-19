#!/usr/bin/env python3
"""Fetch the ONNX cross-encoder used when RERANK is "auto".

The config.RERANK_MODEL repo on Hugging Face (default
cross-encoder/ms-marco-MiniLM-L-6-v2, ~90 MB) ships a pre-exported
onnx/model.onnx, so no torch export is needed:

  .venv/bin/python scripts/fetch_reranker.py

Produces models/reranker-onnx/{model.onnx, tokenizer.json}
(config.ONNX_RERANKER_DIR). Once present, rag._maybe_rerank uses it.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

OUT = Path(config.ONNX_RERANKER_DIR)
BASE = f"https://huggingface.co/{config.RERANK_MODEL}/resolve/main"
FILES = {"model.onnx": f"{BASE}/onnx/model.onnx",
         "tokenizer.json": f"{BASE}/tokenizer.json"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT / name
        if dest.exists():
            print(f"already present: {name}")
            continue
        print(f"downloading {name} …")
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    from encoder import OnnxReranker
    rr = OnnxReranker(OUT)
    q = "How do glaciers carve valleys?"
    good, bad = rr.predict([(q, "Glaciers erode valleys into a U shape as "
                                "ice grinds the bedrock."),
                            (q, "The recipe calls for two cups of flour.")])
    print(f"OnnxReranker OK (relevant {good:.2f} > unrelated {bad:.2f})")
    assert good > bad
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
