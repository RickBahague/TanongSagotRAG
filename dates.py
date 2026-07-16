"""Pipeline stage 1c — per-article date extraction (regex, no LLM).

Reads completed chunk files and writes data/dates/<stem>.json:

  {"_hash": <chunk-file sha256>,
   "<essay>": {"date": "1966-02-06", "precision": "day|month|year|work",
               "method": "title|head|tail|range"}}

Dates live in this sidecar only — chunk bytes, hashes, embeddings, and
summaries are untouched; rag.load_sources attaches dates in memory.
Candidate years are validated against the source's plausible range
(collection source_meta "years", else years in the title/filename) so
quoted history (1896, 1935…) can't win.

  .venv/bin/python dates.py -c <collection> [--force]  (also runs in ingest)
"""
import argparse
import json
import re

import library

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_MON = "|".join(MONTHS)

RX_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MON})\.?,?\s+(\d{{4}})\b", re.I)
RX_MDY = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
RX_MY = re.compile(rf"\b({_MON})\.?,?\s+(\d{{4}})\b", re.I)
RX_Y = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")

DEFAULT_RANGE = (1900, 2100)


def work_years(col, source):
    return col.work_years(source) or DEFAULT_RANGE


# attribution context: a date near one of these is almost certainly the
# article's own date (dateline / source footnote), not discussed history
RX_ATTR = re.compile(r"\b(delivered|published|written|issued|dated|adopted|"
                     r"speech|address|statement|message|lecture|interview|"
                     r"press release)\b", re.I)
_ATTR_WINDOW = 220  # chars around a keyword that count as attribution context
                    # (footnotes: "Speech delivered … before the <long org
                    #  name> at <venue> on December 19, 1965")


def _candidates(text, lo, hi):
    """(precision, iso_date, position, near_attr) within the valid range."""
    attr_spans = [m.start() for m in RX_ATTR.finditer(text)]

    def near(pos):
        # keyword must PRECEDE the date ("delivered … on December 19, 1965");
        # a date before the keyword is being talked about, not attributed
        return any(0 <= pos - a <= _ATTR_WINDOW for a in attr_spans)

    out = []
    for m in RX_DMY.finditer(text):
        d, mon, y = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))
        if lo <= y <= hi and 1 <= d <= 31:
            out.append(("day", f"{y:04d}-{mon:02d}-{d:02d}", m.start(), near(m.start())))
    for m in RX_MDY.finditer(text):
        mon, d, y = MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        if lo <= y <= hi and 1 <= d <= 31:
            out.append(("day", f"{y:04d}-{mon:02d}-{d:02d}", m.start(), near(m.start())))
    for m in RX_MY.finditer(text):
        mon, y = MONTHS[m.group(1).lower()], int(m.group(2))
        if lo <= y <= hi:
            out.append(("month", f"{y:04d}-{mon:02d}", m.start(), near(m.start())))
    for m in RX_Y.finditer(text):
        y = int(m.group(1))
        if lo <= y <= hi:
            out.append(("year", f"{y:04d}", m.start(), near(m.start())))
    return out


def _best(cands):
    """Attribution context beats everything; then precision; then earliest
    position (datelines lead)."""
    rank = {"day": 0, "month": 1, "year": 2}
    return min(cands, key=lambda c: (not c[3], rank[c[0]], c[2])) if cands else None


def article_date(col, essay, chunks, source):
    """(date, precision, method) for one section. Full day/month dates are
    accepted from anywhere in the first chunk (datelines and source footnotes
    sit at the top or the foot of a section's first page); bare years only
    from the tight head window, title, or tail — body text mentions too many
    years to trust them further in."""
    lo, hi = work_years(col, source)
    head80 = " ".join(chunks[0]["text"].split()[:80])
    tail = " ".join(chunks[-1]["text"].split()[-60:])
    for method, text, allow_year in (("title", essay, True),
                                     ("head", head80, True),
                                     ("head", chunks[0]["text"], False),
                                     ("tail", tail, True)):
        cands = _candidates(text, lo, hi)
        if not allow_year:
            cands = [c for c in cands if c[0] != "year"]
        best = _best(cands)
        if best:
            return best[1], best[0], method
    mid = (lo + hi) // 2
    return f"{mid:04d}", "work", "range"


def dates_for_chunk_file(col, chunk_file):
    chunks = [json.loads(l) for l in open(chunk_file)]
    groups = {}
    for c in chunks:
        groups.setdefault(c["essay"], []).append(c)
    out = {}
    for essay, cs in groups.items():
        date, precision, method = article_date(col, essay, cs, cs[0]["source"])
        out[essay] = {"date": date, "precision": precision, "method": method}
    return out


def run_all(col, force=False):
    import ingest
    out_dir = col.dates_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for cf in sorted(col.chunks_dir.glob("*.jsonl")):
        out = out_dir / (cf.stem + ".json")
        h = ingest.file_hash(cf)
        if out.exists() and not force:
            try:
                if json.loads(out.read_text()).get("_hash") == h:
                    continue
            except Exception:
                pass
        d = dates_for_chunk_file(col, cf)
        d["_hash"] = h
        out.write_text(json.dumps(d, ensure_ascii=False))
        n = sum(1 for k in d if k != "_hash")
        print(f"[dates] {cf.stem[:44]}: {n} sections dated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "-c", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_all(library.get(args.collection), args.force)
