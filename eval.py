#!/usr/bin/env python3
"""Retrieval eval for one collection: does top-k contain an expected section?

Questions live per collection in <collection>/data/eval.jsonl:

  {"q": "What is said about X?", "expect": "regex matched against section
   title, work title, or chunk text"}

  .venv/bin/python eval.py -c <collection> [--compare] [-k 10]

--compare scores flat retrieval vs hierarchical side by side (when
summaries exist). Record results in plan.md; keep changes that move the
number (design.md S5: measure, don't guess).
"""
import argparse
import json
import re
import sys

import config
import library
import rag


def load_questions(col):
    f = col.data_dir / "eval.jsonl"
    if not f.exists():
        sys.exit(f"No eval questions at {f}.\nAdd lines like: "
                 '{"q": "…", "expect": "regex"}')
    return [json.loads(l) for l in open(f) if l.strip()]


def hit(index, question, pattern, k):
    rx = re.compile(pattern, re.I)
    for _, c in index.search(question, k):
        if (rx.search(c["essay"]) or rx.search(c.get("work", ""))
                or rx.search(c["text"])):
            return True
    return False


def score(index, questions, k, label):
    n = sum(hit(index, q["q"], q["expect"], k) for q in questions)
    print(f"  {label:14s} {n}/{len(questions)}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "-c", required=True)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--compare", action="store_true",
                    help="flat vs hierarchical")
    args = ap.parse_args()
    col = library.get(args.collection)
    questions = load_questions(col)
    print(f"{col.name}: {len(questions)} questions, top-{args.k}")

    if not args.compare:
        score(rag.load_index(col), questions, args.k, "current")
        return

    sources = rag.load_sources(col)
    chunks = [c for _, cs, _ in sources for c in cs]
    flat = rag.BM25Index(chunks)
    try:
        flat = rag.HybridIndex(sources)
    except Exception:
        pass
    score(flat, questions, args.k, "flat")
    summaries = rag.load_summaries(col)
    if summaries:
        hier = rag.HierarchicalIndex(flat, rag.SummaryIndex(summaries))
        score(hier, questions, args.k, "hierarchical")
    else:
        print("  (no summaries — hierarchical skipped)")


if __name__ == "__main__":
    import errors
    try:
        main()
    except errors.NotIndexed as e:     # collection not ingested yet
        sys.exit(str(e))
