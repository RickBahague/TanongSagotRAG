"""Pipeline Stage 1b — per-section summaries, generated like chunks and
BEFORE any embedding.

Reads ONLY completed chunk files (data/chunks/*.jsonl), asks the configured
local LLM for a 4-6 sentence abstract + keywords per section, and writes
plain-text artifacts to data/summaries/<stem>.jsonl. Stage 2 (embedding)
may embed a summary only after it exists here. Idempotent; --force redoes.

  .venv/bin/python summarize.py -c <collection> [--force]

(Also invoked by ingest.py between its chunk and embed stages.)
"""
import argparse
import json
import re
from datetime import datetime, timezone

import config
import library
import rag

SYSTEM = ("You write neutral, faithful abstracts of documents for a "
          "research index. Never editorialize.")

PROMPT = """Summarize the following text by {author} in 4-6 sentences, \
covering its main arguments and subjects. Then, on one final line, write \
'Keywords:' followed by 5-10 topical keywords separated by commas.

Text: "{title}" (from {work})

{text}"""

WINDOW_PROMPT = """Summarize the following part of a text by {author} in \
3-4 sentences, covering its main arguments and subjects.

Text: "{title}" (part {part})

{text}"""


def article_text(chunks):
    """Continuous article text from its ordered chunks (overlap-trimmed)."""
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP_WORDS
    words, prev = [], None
    for c in sorted(chunks, key=rag._chunk_start):
        w = c["text"].split()
        if prev is not None and rag._chunk_start(c) - rag._chunk_start(prev) == step:
            w = w[config.CHUNK_OVERLAP_WORDS:]
        words.extend(w)
        prev = c
    return words


def parse_reply(reply):
    """Split an LLM reply into (summary, keywords)."""
    kw = ""
    m = re.search(r"^\s*\**\s*keywords\s*:?\**\s*(.+)$", reply,
                  re.I | re.M)
    if m:
        kw = m.group(1).strip().rstrip(".")
        reply = reply[:m.start()].strip()
    return re.sub(r"\s+", " ", reply).strip(), kw


def summarize_article(col, title, work, words):
    """One section -> (summary, keywords), map-reducing long sections."""
    author = col.author_label
    if len(words) <= config.SUMMARY_MAX_WORDS:
        text = " ".join(words)
    else:
        parts = []
        for n, i in enumerate(range(0, len(words), config.SUMMARY_MAX_WORDS), 1):
            wtext = " ".join(words[i:i + config.SUMMARY_MAX_WORDS])
            parts.append(rag.llm_chat(SYSTEM, WINDOW_PROMPT.format(
                author=author, title=title, part=n, text=wtext)))
        text = "Part summaries of the text:\n" + "\n".join(parts)
    reply = rag.llm_chat(SYSTEM, PROMPT.format(
        author=author, title=title, work=work, text=text))
    return parse_reply(reply)


def _workers():
    """Concurrent LLM calls; the in-process embedded backend is not
    thread-safe, so it is always driven serially."""
    if config.LLM_BACKEND == "embedded":
        return 1
    return max(config.SUMMARY_WORKERS, 1)


def summarize_source(col, chunk_file, force=False):
    """Stage 1b for one already-chunked source. Returns n_sections or None
    if up to date.

    Staleness is content-based (sidecar hash of the chunk file), so an
    identical re-chunk costs nothing. Sections run through a thread pool
    (SUMMARY_WORKERS) and each completed section is APPENDED to the JSONL
    immediately — an interrupted run resumes at the next unfinished section
    (matched by title); the sidecar is written only when the whole source
    is complete."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import ingest
    out_dir = col.summaries_dir
    out = out_dir / chunk_file.name
    sidecar = out.with_suffix(".jsonl.src")
    chunk_hash = ingest.file_hash(chunk_file)
    if (out.exists() and not force and sidecar.exists()
            and sidecar.read_text().strip() == chunk_hash):
        return None

    chunks = [json.loads(l) for l in open(chunk_file)]
    groups = {}
    for c in chunks:
        groups.setdefault(c["essay"], []).append(c)

    # resume: the sidecar marks "<hash> partial" while a run is in flight
    # and plain "<hash>" once complete. A partial file for the SAME chunk
    # hash resumes (skip finished articles); anything else starts clean.
    done = set()
    prior = sidecar.read_text().strip() if sidecar.exists() else ""
    if (out.exists() and not force
            and prior == f"{chunk_hash} partial"):
        done = {json.loads(l)["essay"] for l in open(out) if l.strip()}
        print(f"    resuming: {len(done)} articles already done")
    elif out.exists():
        out.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(f"{chunk_hash} partial")

    todo = []
    for title, cs in groups.items():
        words = article_text(cs)
        if len(words) < config.SUMMARY_MIN_WORDS or title in done:
            continue
        todo.append((title, cs, words))

    lock = threading.Lock()
    n = len(done)

    def run_one(job):
        nonlocal n
        title, cs, words = job
        work = col.work_title(cs[0]["source"])
        summary, keywords = summarize_article(col, title, work, words)
        rec = json.dumps({
            "essay": title, "source": cs[0]["source"], "summary": summary,
            "keywords": keywords,
            "page_start": min(c["page_start"] for c in cs),
            "page_end": max(c["page_end"] for c in cs),
        }, ensure_ascii=False)
        with lock:                       # checkpoint: one line per article
            with open(out, "a") as f:
                f.write(rec + "\n")
            n += 1
        print(f"    {title[:60]} ({len(words)} words) ok")

    if todo:
        with ThreadPoolExecutor(max_workers=min(_workers(), len(todo))) as pool:
            list(pool.map(run_one, todo))
    sidecar.write_text(chunk_hash)       # source complete -> mark fresh
    return n


def llm_reachable():
    try:
        rag.llm_chat("Reply with the single word: ok", "ok?")
        return True
    except Exception:
        return False


def _manifest_key(col, stem):
    """Manifest entries are keyed by source FILENAME; map a chunk-file stem
    back to it (sources may be .pdf/.md/.docx/…)."""
    for src in col.source_files():
        if src.stem == stem:
            return src.name
    return stem


def run_all(col, force=False, manifest=None, save=None, progress=None):
    """Stage 1b over every completed chunk file of one collection."""
    if not llm_reachable():
        print("[summarize] LLM backend unreachable — stage skipped "
              "(embedding will proceed with chunks only)")
        return False
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for chunk_file in sorted(col.chunks_dir.glob("*.jsonl")):
        try:
            n = summarize_source(col, chunk_file, force)
        except Exception as e:
            print(f"[summarize] {chunk_file.stem}: interrupted ({e}) — "
                  f"partial progress kept, will resume next run")
            continue
        status = "up to date" if n is None else f"{n} section summaries"
        if progress:
            progress({"stage": "summarize", "source": chunk_file.stem,
                      "status": status})
        print(f"[summarize] {chunk_file.stem}: {status}")
        if manifest is not None:
            m = manifest.setdefault(_manifest_key(col, chunk_file.stem), {})
            m["summary_status"] = status
            if n is not None:
                m["summarized_at"] = now
            if save:
                save(manifest)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "-c", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import ingest
    col = library.get(args.collection)
    manifest = ingest.load_manifest(col)
    run_all(col, args.force, manifest, lambda m: ingest.save_manifest(col, m))
