#!/usr/bin/env python3
"""Retrieval + gate eval for one collection.

Questions live per collection in <collection>/data/eval.jsonl, one JSON
object per line:

  {"q": "What is said about X?", "expect": "regex"}   # on-topic
  {"q": "Capital of Mongolia?",  "expect": null}      # off-topic

`expect` is matched case-insensitively (re.search) against a hit's section
title (`essay`), work title (`work`) or chunk `text`; a hit "matches" if any
of the three does.

  .venv/bin/python eval.py -c <collection> [-k 10] [--json] [--variants]

Metrics (on-topic questions only; null-expect lines never count as misses):
  hit@1, hit@3, hit@k  fraction of on-topic questions with >= 1 matching hit
                       within the top 1 / 3 / k results.
  recall@k             per question: (# matching chunks in the top k) /
                       (# chunks in the WHOLE collection that match the
                       regex), averaged over on-topic questions. A question
                       whose regex matches no chunk at all scores 0.
  mrr                  mean of 1/rank of the first matching hit; 0 when no
                       match within the top k.
Gate (rag.gate_ok, computed with the same index as the metrics):
  false_accepts        off-topic questions the gate lets through.
  false_rejects        on-topic questions the gate blocks.

--variants evaluates bm25, hybrid (vectors + query encoder), hierarchical
(summaries, over the best flat base) and +rerank (cross-encoder importable;
over the best preceding variant); each missing one is reported as
"unavailable: <reason>". --compare is an alias kept for old habits.
--json prints one machine-readable object. Record results in plan.md; keep
changes that move the number (design.md S5: measure, don't guess).
"""
import argparse
import contextlib
import importlib.util
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
                 '{"q": "…", "expect": "regex"}  (or "expect": null for '
                 "an off-topic question the gate should reject)")
    return [json.loads(l) for l in open(f) if l.strip()]


def matches(rx, c):
    return bool(rx.search(c.get("essay", "")) or rx.search(c.get("work", ""))
                or rx.search(c.get("text", "")))


# ------------------------------------------------------------- variants --

class RerankIndex:
    """Wraps an index: fetch a wider pool, reorder with rag._maybe_rerank.
    Exposes the base's bm25/dense/chunks so rag.gate_ok sees the same
    signals the base would."""

    def __init__(self, base):
        self.base = base
        self.chunks = base.chunks
        self.bm25 = base if isinstance(base, rag.BM25Index) else getattr(base, "bm25", None)
        self.dense = base if isinstance(base, rag.DenseIndex) else getattr(base, "dense", None)

    def search(self, query, k=config.TOP_K):
        pool = self.base.search(query, k * getattr(config, "POOL_FACTOR", 3))
        return rag._maybe_rerank(query, pool)[:k]


def _rerank_available():
    """None if the cross-encoder loads, else a reason string."""
    if importlib.util.find_spec("sentence_transformers") is None:
        return "sentence_transformers not installed"
    if config.RERANK == "off":
        return "RERANK is off in config"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            rag._maybe_rerank("probe", [(1.0, {"text": "probe"})])
    except Exception as e:
        return f"reranker failed ({e})"
    if not getattr(rag, "_RERANKER", None):
        return "cross-encoder model could not be loaded"
    return None


def describe(index):
    name = type(index).__name__
    base = getattr(index, "base", None)
    return f"{name}({describe(base)})" if base is not None else name


# ------------------------------------------------------------- scoring --

def evaluate(index, questions, all_chunks, k):
    on = [q for q in questions if q.get("expect")]
    off = [q for q in questions if not q.get("expect")]
    h1 = h3 = hk = 0
    rr = rec = 0.0
    misses = []
    for q in on:
        rx = re.compile(q["expect"], re.I)
        hits = index.search(q["q"], k)[:k]
        flags = [matches(rx, c) for _, c in hits]
        rank = next((i for i, f in enumerate(flags, 1) if f), None)
        if rank:
            h1 += rank <= 1
            h3 += rank <= 3
            hk += 1
            rr += 1.0 / rank
        else:
            misses.append(q["q"])
        relevant = {c["id"] for c in all_chunks if matches(rx, c)}
        got = {c["id"] for (_, c), f in zip(hits, flags) if f}
        rec += len(got & relevant) / len(relevant) if relevant else 0.0
    n = len(on) or 1
    fr = [q["q"] for q in on if not rag.gate_ok(index, q["q"])]
    fa = [q["q"] for q in off if rag.gate_ok(index, q["q"])]
    return {
        "retrieval": describe(index),
        "questions_on_topic": len(on),
        "questions_off_topic": len(off),
        "hit@1": h1 / n, "hit@3": h3 / n, f"hit@{k}": hk / n,
        f"recall@{k}": rec / n, "mrr": rr / n,
        "misses": misses,
        "gate_false_accepts": len(fa), "gate_false_rejects": len(fr),
        "false_accepts": fa, "false_rejects": fr,
    }


def run_variants(col, questions, k):
    """{name: metrics | "unavailable: <reason>"}"""
    sources = rag.load_sources(col)
    chunks = [c for _, cs, _ in sources for c in cs]
    out, built = {}, {}

    def add(name, make):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                idx = make()
            out[name] = evaluate(idx, questions, chunks, k)
            built[name] = idx
        except Exception as e:
            out[name] = f"unavailable: {type(e).__name__}: {e}"

    add("bm25", lambda: rag.BM25Index(chunks))
    if not any(v for _, _, v in sources):
        out["hybrid"] = "unavailable: no vector files (ingest with embedding deps)"
    else:
        add("hybrid", lambda: rag.HybridIndex(sources))

    def best(names):
        avail = [n for n in names if n in built]
        return max(avail, key=lambda n: (out[n]["mrr"], out[n][f"hit@{k}"],
                                         names.index(n)))

    flat = best(["bm25", "hybrid"])
    summaries = rag.load_summaries(col)
    if not summaries:
        out["hierarchical"] = "unavailable: no summaries (run summarize.py)"
    else:
        add("hierarchical", lambda: rag.HierarchicalIndex(
            built[flat], rag.SummaryIndex(summaries)))

    reason = _rerank_available()
    if reason:
        out["+rerank"] = f"unavailable: {reason}"
    else:
        top = best(["bm25", "hybrid", "hierarchical"])
        add("+rerank", lambda: RerankIndex(built[top]))
    return out


# --------------------------------------------------------------- output --

def _row(name, m, k):
    if isinstance(m, str):
        return f"  {name:13s} {m}"
    return (f"  {name:13s} {m['hit@1']:5.2f} {m['hit@3']:5.2f} "
            f"{m[f'hit@{k}']:6.2f} {m[f'recall@{k}']:6.2f} {m['mrr']:5.2f} "
            f"{m['gate_false_accepts']:3d} {m['gate_false_rejects']:3d}")


def print_table(col, k, results):
    first = next((m for m in results.values() if isinstance(m, dict)), None)
    on = first["questions_on_topic"] if first else 0
    off = first["questions_off_topic"] if first else 0
    print(f"{col.name}: {on} on-topic + {off} off-topic questions, top-{k}")
    print(f"  {'':13s} {'hit@1':>5s} {'hit@3':>5s} {f'hit@{k}':>6s} "
          f"{f'rec@{k}':>6s} {'mrr':>5s} {'FA':>3s} {'FR':>3s}")
    for name, m in results.items():
        print(_row(name, m, k))
    for name, m in results.items():
        if not isinstance(m, dict):
            continue
        tag = f" [{name}]" if len(results) > 1 else ""
        for q in m["misses"]:
            print(f"  miss{tag}: {q}")
        for q in m["false_accepts"]:
            print(f"  false-accept{tag}: {q}")
        for q in m["false_rejects"]:
            print(f"  false-reject{tag}: {q}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", "-c", required=True)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--json", action="store_true",
                    help="print one machine-readable JSON object")
    ap.add_argument("--variants", "--compare", action="store_true",
                    help="compare bm25 / hybrid / hierarchical / +rerank")
    args = ap.parse_args(argv)
    col = library.get(args.collection)
    questions = load_questions(col)

    if args.variants:
        results = run_variants(col, questions, args.k)
        retrieval = None
    else:
        with contextlib.redirect_stdout(sys.stderr):   # keep --json clean
            index = rag.load_index(col)
        chunks = rag.load_chunks(col)
        results = {"current": evaluate(index, questions, chunks, args.k)}
        retrieval = results["current"]["retrieval"]

    if args.json:
        doc = {"collection": col.id, "k": args.k,
               "n_questions": len(questions)}
        if args.variants:
            doc["variants"] = results
        else:
            doc.update(results["current"])
            doc["retrieval"] = retrieval
        print(json.dumps(doc, indent=2))
    else:
        print_table(col, args.k, results)
    return results


if __name__ == "__main__":
    import errors
    try:
        main()
    except errors.NotIndexed as e:     # collection not ingested yet
        sys.exit(str(e))
