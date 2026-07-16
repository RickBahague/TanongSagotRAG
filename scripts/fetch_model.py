#!/usr/bin/env python3
"""Download the bundled GGUF model (run on the host before packaging).

  .venv/bin/python scripts/fetch_model.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

URL = ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
       "qwen2.5-1.5b-instruct-q4_k_m.gguf")


def main():
    dest = Path(config.LLM_MODEL_PATH)
    if dest.exists():
        print(f"Already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n -> {dest}")

    def hook(n, size, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {n*size/1e6:,.0f} / {total/1e6:,.0f} MB", end="")
    urllib.request.urlretrieve(URL, dest, reporthook=hook)
    print(f"\nDone: {dest.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
