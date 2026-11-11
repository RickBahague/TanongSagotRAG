#!/usr/bin/env python3
"""Retrieval + gate numbers on the fixture corpus, as JSON.

Independent of eval.py on purpose: this is the yardstick scripts/verify.sh
uses to compare milestones, so it must not change when eval.py does.

  .venv/bin/python tests/eval_fixture.py [--out FILE]

Runs in a throwaway ANYRAG_HOME (set below, before project imports) over
the clean fixture files, with whatever retrieval rag.load_index() picks
(BM25 today; hybrid once vectors are produced). Never touches devdata/.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME = Path(tempfile.mkdtemp(prefix="anyrag-eval-"))
os.environ["ANYRAG_HOME"] = str(HOME)
sys.path[:0] = [str(REPO), str(REPO / "tests")]

import config  # noqa: E402
import ingest  # noqa: E402
import library  # noqa: E402
import rag  # noqa: E402
from helpers import CLEAN_FILES, CORPUS, load_eval  # noqa: E402

K = 10


def _rank(hits, pattern):
    rx = re.compile(pattern, re.I)
    for i, (_, c) in enumerate(hits, start=1):
        if (rx.search(c["essay"]) or rx.search(c.get("work", ""))
                or rx.search(c["text"])):
            return i
    return None


def run():
    assert HOME in Path(config.USER_ROOT).resolve().parents or \
        Path(config.USER_ROOT).resolve() == HOME.resolve()
    docs = HOME / "docs"
    docs.mkdir()
    for f in CLEAN_FILES:
        shutil.copy2(CORPUS / f, docs / f)
    col = library.create("Eval fixtures", link=str(docs))
    ingest.run_pipeline(col, summaries=False, workers=1,
                        progress=lambda e: None)
    col = library.get(col.id)
    index = rag.load_index(col)

    qs = load_eval()
    on = [q for q in qs if q["expect"]]
    off = [q for q in qs if not q["expect"]]
    ranks = [_rank(index.search(q["q"], K), q["expect"]) for q in on]
    false_rejects = [q["q"] for q in on if not rag.gate_ok(index, q["q"])]
    false_accepts = [q["q"] for q in off if rag.gate_ok(index, q["q"])]
    n = len(on)
    vec_files = sorted(col.vectors_dir.glob("*.npz"))
    vec_keys = set()
    for f in vec_files:
        try:
            import numpy as np
            vec_keys |= set(np.load(f).files)
        except Exception:
            pass
    return {
        "vector_files": len(vec_files),
        "vector_keys": sorted(vec_keys),
        "retrieval": type(index).__name__,
        "chunks": len(index.chunks),
        "questions_on_topic": n,
        "questions_off_topic": len(off),
        "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        f"hit@{K}": sum(1 for r in ranks if r) / n,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
        "gate_false_accepts": len(false_accepts),
        "gate_false_rejects": len(false_rejects),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    try:
        res = run()
    finally:
        shutil.rmtree(HOME, ignore_errors=True)
    text = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
