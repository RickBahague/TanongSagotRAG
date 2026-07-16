"""Pipeline stage 1d — quotable-passage bank (heuristic, no LLM).

Scores 1-3 sentence windows from every chunk for self-containedness and
aphoristic quality; keeps the best few per article. Every entry's text is a
VERBATIM substring of its chunk, so quotes are verifiable by construction.

Output: data/quotes/<stem>.jsonl  {text, chunk_id, essay, source, page, score}
with the usual hash-idempotence ("_hash" line first).

  .venv/bin/python quotes_bank.py -c <collection> [--force]  (also in ingest)
"""
import argparse
import json
import re

import library

MIN_WORDS, MAX_WORDS = 15, 70
PER_ARTICLE = 5

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BAD_START = re.compile(
    r"^(but|and|so|thus|therefore|however|yet|also|moreover|furthermore|"
    r"nevertheless|hence|besides|then|still|instead|likewise|meanwhile|"
    r"he|she|they|it|this|that|these|those|such|its|his|her|their|"
    r"i wish|i thank|i am glad|i am deeply|thank)\b", re.I)
_BOILER = re.compile(r"\b(congratulat|greeting|felicitat|anniversar|"
                     r"condolence|birthday|thank you|long live)\b", re.I)
_APHORISM = re.compile(r"\b(must|cannot|can never|is the|are the|there is no|"
                       r"no amount of|only the|it is necessary|shall)\b", re.I)


def _sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def score_window(text):
    """Score a candidate passage; None = disqualified."""
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return None
    if not text[0].isupper() and text[0] not in "“\"'":
        return None
    if text[-1] not in ".!?”\"'":
        return None
    if _BAD_START.match(text):
        return None
    if "[" in text or "]" in text or "…" in text or "..." in text:
        return None
    if " - " in text:                             # broken hyphenation artifact
        return None
    digits = sum(ch.isdigit() for ch in text)
    if digits > len(text) * 0.03:                 # page furniture / footnotes
        return None
    if _BOILER.search(text):
        return None
    score = 1.0
    score += 2.0 * len(_APHORISM.findall(text))
    if 20 <= len(words) <= 45:                    # sweet spot
        score += 1.0
    starts = [s.split()[0].lower() for s in _sentences(text) if s.split()]
    if len(starts) > 1 and len(set(starts)) == 1:  # parallelism
        score += 1.5
    return score


def candidates_for_chunk(c):
    sents = _sentences(c["text"])
    # drop first/last sentence of a chunk: likely cut mid-thought by chunking
    sents = sents[1:-1]
    out = []
    for i in range(len(sents)):
        for n in (1, 2, 3):
            if i + n > len(sents):
                break
            text = " ".join(sents[i:i + n])
            s = score_window(text)
            if s is not None and text in c["text"]:   # verbatim by construction
                out.append((s, text))
    return out


def bank_for_chunk_file(chunk_file):
    import rag
    chunks = rag.dedup_ids([json.loads(l) for l in open(chunk_file)])
    groups = {}
    for c in chunks:
        groups.setdefault(c["essay"], []).append(c)
    entries = []
    for essay, cs in groups.items():
        cands = []
        for c in cs:
            for s, text in candidates_for_chunk(c):
                cands.append((s, text, c))
        cands.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for s, text, c in cands:
            key = re.sub(r"\W+", "", text.lower())[:80]
            if key in seen:                        # chunk-overlap duplicates
                continue
            seen.add(key)
            picked.append({"text": text, "chunk_id": c["id"], "essay": essay,
                           "source": c["source"], "page": c["page_start"],
                           "score": round(s, 2)})
            if len(picked) >= PER_ARTICLE:
                break
        entries.extend(picked)
    return entries


def run_all(col, force=False):
    import ingest
    out_dir = col.quotes_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for cf in sorted(col.chunks_dir.glob("*.jsonl")):
        out = out_dir / cf.name
        h = ingest.file_hash(cf)
        if out.exists() and not force:
            try:
                first = json.loads(open(out).readline())
                if first.get("_hash") == h:
                    continue
            except Exception:
                pass
        entries = bank_for_chunk_file(cf)
        with open(out, "w") as f:
            f.write(json.dumps({"_hash": h}) + "\n")
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[quotes] {cf.stem[:44]}: {len(entries)} quotable passages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "-c", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_all(library.get(args.collection), args.force)
