"""Staged ingestion pipeline. For EVERY new source in a collection:

  Stage 1  — CHUNK:     extract + segment + clean + chunk
                        → data/chunks/<source>.jsonl  (+ paras sidecar)
  Stage 1c — DATES      (regex sidecar, dates.py)
  Stage 1d — QUOTE BANK (heuristic sidecar, quotes_bank.py)
  Stage 1b — SUMMARIZE  (local LLM, summarize.py; skipped if unreachable)
  Stage 2  — EMBED:     chunks AND summaries → data/vectors/*.npz
                        (optional; only artifacts that already exist as text)

Chunking is always completed for a source before any embedding of it.
data/manifest.json records pipeline state per source; unchanged sources are
skipped (content-hash staleness for vectors/summaries, mtime for chunks).

  .venv/bin/python ingest.py --collection <id> [--force] [--no-summaries]

Supported sources: .pdf (pypdf), .txt, .md, .docx (python-docx), .html/.htm

Section segmentation ladder (PDF) — best rung wins, rung recorded in the
manifest per source:
  meta     explicit toc_pdf_pages/page_offset in the collection's source_meta
  outline  the PDF's own bookmarks/outline
  toc      numbered-TOC parse with printed-page offset detection
  match    TOC titles matched against page heads (ebook-style)
  none     whole file as one section
.md segments on #/## headings; .docx on Heading styles; .txt/.html are
single-section.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
import library


# ---------------------------------------------------------------- manifest --

def load_manifest(col):
    if col.manifest_file.exists():
        return json.loads(col.manifest_file.read_text())
    return {}


def save_manifest(col, m):
    col.data_dir.mkdir(parents=True, exist_ok=True)
    col.manifest_file.write_text(json.dumps(m, indent=2))


# ------------------------------------------------------- stage 1: chunking --

def parse_toc(reader, toc_pdf_pages):
    """Parse a table of contents -> [(title, printed_start_page)]."""
    text = ""
    for i in toc_pdf_pages:
        text += (reader.pages[i].extract_text() or "") + "\n"
    entries = []
    for line in text.splitlines():
        m = re.match(r"^(.*?)\s*─*[\s…\.�●]*?(\d+|[ivxlc]+)\s*$", line.strip())
        if not m:
            continue
        title = re.sub(r"[\s…\.●]+$", "", m.group(1).replace("�", " ")).strip()
        title = re.sub(r"\s+", " ", title)
        page = m.group(2)
        if title and title.upper() != "CONTENTS" and page.isdigit():
            entries.append((title, int(page)))
    return entries


def essay_for_page(toc, printed_page):
    current = "Front matter"
    for title, start in toc:
        if printed_page >= start:
            current = title
        else:
            break
    return current


def clean_page(text, printed_page, book_title):
    """Strip running headers/page numbers, fix hyphenation artifacts."""
    lines = text.splitlines()
    if lines and lines[0].strip().isdigit():
        lines = lines[1:]
    if lines and len(lines[0].strip()) < 70:
        header = lines[0].strip()
        if header == book_title or (printed_page % 2 == 1 and not header.isupper()
                                    and re.match(r"^[A-Z][\w\s'’?.,-]+$", header)
                                    and len(header.split()) <= 8):
            lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_words(words, size, overlap):
    step = size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        yield start, words[start:start + size]


_PARA_END = tuple('.!?”")»')


def _is_para_break(prev_line, line):
    """Layout heuristic: does `line` start a new paragraph after `prev_line`?"""
    if prev_line == "":                      # blank line
        return True
    if prev_line.isupper():                  # heading line
        return True
    if (prev_line.endswith(_PARA_END) and len(prev_line) < 55
            and line[:1] and (line[0].isupper() or line[0] in "“‘\"'—–0123456789")):
        return True                          # short sentence-final line
    return False


def words_with_paras(pages):
    """Word list + page labels + set of word offsets starting a paragraph.
    Produces the exact same word sequence as `text.split()` per page, so
    chunk output stays byte-identical — paragraphs are a display-only
    sidecar (see design.md)."""
    words, word_pages, paras = [], [], set()
    prev = None
    for label, text in pages:
        for raw in text.splitlines():
            ls = raw.strip()
            if not ls:
                prev = ""
                continue
            if prev is not None and words and _is_para_break(prev, ls):
                paras.add(len(words))
            ws = ls.split()
            words.extend(ws)
            word_pages.extend([label] * len(ws))
            prev = ls
        # page boundary: paragraph may continue — keep prev across pages
    return words, word_pages, paras


# -------------------------------------------- automatic section detection --

_ROMAN_SUB = re.compile(r"^\s*[IVXLC]+\.\s")      # 'I. …' subsection lines
_TOC_LINE = re.compile(r"^(.{3,}?)[\s.…�●─]*?(\d{1,4})\s*$")


def _detect_offset(reader):
    """printed-page → pdf-page offset, from pages starting with a bare number."""
    from collections import Counter
    votes = Counter()
    for i in range(4, min(len(reader.pages), 80)):
        lines = (reader.pages[i].extract_text() or "").splitlines()
        first = lines[0].strip() if lines else ""
        if first.isdigit() and 0 < int(first) <= i:
            votes[i - int(first)] += 1
    return votes.most_common(1)[0][0] if votes else None


def _toc_text(reader, max_scan=18):
    """Concatenated text of pages that look like a table of contents."""
    parts = []
    for i in range(min(len(reader.pages), max_scan)):
        t = reader.pages[i].extract_text() or ""
        numbered = sum(1 for l in t.splitlines() if _TOC_LINE.match(l.strip()))
        if "contents" in t[:400].lower() or numbered >= 6:
            parts.append(t)
    return "\n".join(parts)


def _parse_numbered_toc(text):
    """[(title, printed_page)] from a page-numbered TOC. Wrapped entries
    (title spanning lines, page number on the last one) are merged via a
    line buffer; roman-numbered subsections are dropped after merging."""
    entries, last, buf = [], 0, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            buf = []
            continue
        m = _TOC_LINE.match(line)
        if not m:
            if len(line) >= 4 and not line.isdigit():
                buf.append(line)          # possible wrapped-title fragment
                if len(buf) > 2:          # not a TOC region after all
                    buf = buf[-2:]
            continue
        title = " ".join(buf + [m.group(1)])
        buf = []
        title = title.replace("�", " ")     # PDF-extraction dot leaders
        title = re.sub(r"[\s…\.●─]+$", "", title).strip()
        title = re.sub(r"\s+", " ", title)
        page = int(m.group(2))
        if (4 <= len(title) <= 120 and title.lower() != "contents"
                and not _ROMAN_SUB.match(title)
                and page >= last and page < 3000):
            entries.append((title, page))
            last = page
    return entries if len(entries) >= 3 else []


def _toc_titles_only(text):
    """Title lines from a TOC without page numbers (ebook-style)."""
    skip = {"table of contents", "contents", "title page"}
    titles = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line.strip())
        if len(line) >= 15 and not _TOC_LINE.match(line) \
                and line.lower() not in skip \
                and not line.replace(" ", "").isdigit():
            titles.append(line)
    return titles


def _norm_head(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _match_titles_to_pages(reader, titles):
    """Locate section starts by matching TOC titles against page heads.
    Returns [(title, pdf_page)] in page order."""
    heads = []
    for i in range(len(reader.pages)):
        t = reader.pages[i].extract_text() or ""
        lines = [l for l in t.splitlines() if l.strip()][:4]
        heads.append(_norm_head(" ".join(lines))[:160])
    bounds, cursor = [], 0
    for title in titles:
        key = _norm_head(title)[:40]
        if len(key) < 12:
            continue
        for i in range(cursor + 1, len(heads)):
            if heads[i].startswith(key) or key in heads[i][:80]:
                bounds.append((title, i))
                cursor = i
                break
    return bounds if len(bounds) >= 3 else []


def _outline_sections(reader):
    """Segmentation from the PDF's own outline/bookmarks (ladder rung 2).
    Top-level entries only; returns [(title, start_pdf_page, None)] or None."""
    try:
        outline = reader.outline
    except Exception:
        return None
    bounds = []
    for item in outline or []:
        if isinstance(item, list):        # nested children — skip (top level only)
            continue
        try:
            title = re.sub(r"\s+", " ", str(item.title or "")).strip()
            page = reader.get_destination_page_number(item)
        except Exception:
            continue
        if title and page is not None and len(title) >= 3:
            bounds.append((title, page))
    bounds = [(t, p) for t, p in bounds if p < len(reader.pages)]
    bounds.sort(key=lambda b: b[1])
    # drop duplicate pages (multiple bookmarks on one page keep the first)
    seen, out = set(), []
    for t, p in bounds:
        if p not in seen:
            seen.add(p)
            out.append((t, p, None))
    return out if len(out) >= 3 else None


def _auto_sections(reader):
    """Section segmentation without explicit config (ladder rungs toc/match).
    Returns (rung, [(title, start_pdf_page, page_label_offset_or_None)]) or
    (None, None)."""
    toc = _toc_text(reader)
    if not toc:
        return None, None
    offset = _detect_offset(reader)
    numbered = _parse_numbered_toc(toc)
    if numbered and offset is not None:
        return "toc", [(t, p + offset, offset) for t, p in numbered
                       if p + offset < len(reader.pages)]
    # no reliable page offset: locate sections by matching titles to page
    # heads — try numbered-TOC titles first, then ebook-style title lines
    for titles in ([t for t, _ in numbered], _toc_titles_only(toc)):
        bounds = _match_titles_to_pages(reader, titles)
        if bounds:
            return "match", [(t, i, offset) for t, i in bounds]
    return None, None


def sections_from_pdf(col, path, meta):
    """(rung, generator of (section_title, [(page_label, text), ...]))."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    work = col.work_title(path.name)

    if meta and "toc_pdf_pages" in meta:  # known book: TOC + printed-page offset
        toc = parse_toc(reader, meta["toc_pdf_pages"])
        offset, title = meta.get("page_offset", 0), meta.get("title", work)

        def gen_meta():
            sections = {}
            for i in range(offset + 1, len(reader.pages)):
                printed = i - offset
                raw = reader.pages[i].extract_text() or ""
                if raw.strip():
                    essay = essay_for_page(toc, printed)
                    sections.setdefault(essay, []).append(
                        (printed, clean_page(raw, printed, title)))
            yield from sections.items()
        return "meta", gen_meta()

    bounds = _outline_sections(reader)
    rung = "outline" if bounds else None
    if not bounds:
        rung, bounds = _auto_sections(reader)

    if bounds:                            # sections found automatically

        def gen_auto():
            sections = {}
            offset = bounds[0][2]
            for i in range(len(reader.pages)):
                raw = reader.pages[i].extract_text() or ""
                if not raw.strip():
                    continue
                title = work              # front matter before first section
                for t, start, _ in bounds:
                    if i >= start:
                        title = t
                    else:
                        break
                label = i - offset if (offset is not None and i > offset) else i + 1
                sections.setdefault(title, []).append(
                    (label, clean_page(raw, label, work)))
            yield from sections.items()
        return rung, gen_auto()

    # last resort: one section, pdf page numbers
    def gen_flat():
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            if raw.strip():
                pages.append((i, re.sub(r"[ \t]+", " ", raw).strip()))
        yield work, pages
    return "none", gen_flat()


_MD_HEADING = re.compile(r"^#{1,2}\s+(.+?)\s*#*\s*$")


def sections_from_markdown(path):
    """Segment a .md file on #/## headings (≥2 headings, else single)."""
    text = path.read_text(errors="replace")
    sections, current, buf = [], None, []
    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            if buf and (current or "".join(buf).strip()):
                sections.append((current or path.stem, "\n".join(buf)))
            current, buf = m.group(1), []
        else:
            buf.append(line)
    sections.append((current or path.stem, "\n".join(buf)))
    named = [s for s in sections if s[0] != path.stem]
    if len(named) < 2:
        return "none", iter([(path.stem, [(1, text)])])
    return "headings", iter([(t, [(1, body)]) for t, body in sections
                             if body.strip()])


def sections_from_docx(path):
    """Segment a .docx on Heading 1/Title styles (python-docx)."""
    try:
        import docx
    except ImportError:
        raise RuntimeError(".docx import needs python-docx "
                           "(pip install python-docx)")
    d = docx.Document(str(path))
    sections, current, buf = [], None, []
    for p in d.paragraphs:
        style = (p.style.name or "") if p.style else ""
        text = p.text.strip()
        if style.startswith(("Heading 1", "Title")) and text:
            if buf:
                sections.append((current or path.stem, "\n".join(buf)))
            current, buf = text, []
        elif text:
            buf.append(p.text)
        else:
            buf.append("")               # blank line = paragraph break signal
    sections.append((current or path.stem, "\n".join(buf)))
    named = [s for s in sections if s[0] != path.stem and s[1].strip()]
    if len(named) < 2:
        whole = "\n".join(b for _, b in sections)
        return "none", iter([(path.stem, [(1, whole)])])
    return "headings", iter([(t, [(1, body)]) for t, body in sections
                             if body.strip()])


from html.parser import HTMLParser


class _HTMLText(HTMLParser):
    """Extract readable text; <h1>/<h2> become heading lines, <p>/<br>/<div>
    boundaries become blank lines (paragraph signals for words_with_paras)."""
    _SKIP = {"script", "style", "noscript", "head"}
    _BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
              "blockquote", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip, self.title = [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BREAK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BREAK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self):
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()


def sections_from_html(path):
    p = _HTMLText()
    p.feed(path.read_text(errors="replace"))
    title = re.sub(r"\s+", " ", p.title).strip() or path.stem
    return "none", iter([(title, [(1, p.text())])])


def sections_from_text(path):
    text = path.read_text(errors="replace")
    return "none", iter([(path.stem, [(1, text)])])


def extract_sections(col, path):
    """Dispatch on suffix → (rung, sections generator)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return sections_from_pdf(col, path, col.source_meta.get(path.name))
    if suffix == ".md":
        return sections_from_markdown(path)
    if suffix == ".docx":
        return sections_from_docx(path)
    if suffix in (".html", ".htm"):
        return sections_from_html(path)
    return sections_from_text(path)


def chunk_source(col, path, force=False):
    """Stage 1 for one source. Also emits a paragraph-map sidecar
    (data/paras/<stem>.json: chunk_id -> in-chunk word offsets starting a
    paragraph) used only for display; chunk bytes are unaffected.
    Returns (chunk_file, n_chunks, segmentation_rung) or None if skipped."""
    out = col.chunks_dir / (path.stem + ".jsonl")
    para_out = col.paras_dir / (path.stem + ".json")
    if (out.exists() and para_out.exists() and not force
            and out.stat().st_mtime >= path.stat().st_mtime):
        return None
    rung, sections = extract_sections(col, path)

    col.chunks_dir.mkdir(parents=True, exist_ok=True)
    col.paras_dir.mkdir(parents=True, exist_ok=True)
    n, para_map = 0, {}
    chunk_w = col.setting("CHUNK_WORDS", config.CHUNK_WORDS)
    overlap_w = col.setting("CHUNK_OVERLAP_WORDS", config.CHUNK_OVERLAP_WORDS)
    with open(out, "w") as f:
        for title, pages in sections:
            words, word_pages, paras = words_with_paras(pages)
            for start, ck in chunk_words(words, chunk_w, overlap_w):
                if len(ck) < 40:
                    continue
                cid = f"{path.stem[:30]}-{title[:40]}-{start}"
                f.write(json.dumps({
                    "id": cid,
                    "text": " ".join(ck),
                    "essay": title,
                    "page_start": word_pages[start],
                    "page_end": word_pages[min(start + len(ck) - 1,
                                               len(word_pages) - 1)],
                    "source": path.name,
                }, ensure_ascii=False) + "\n")
                rel = sorted(p - start for p in paras if start < p < start + len(ck))
                if rel:
                    para_map[cid] = rel
                n += 1
    para_out.write_text(json.dumps(para_map))
    return out, n, rung


# ---------------------------------------------------- metadata auto-detect --

def _prettify(stem):
    s = re.sub(r"[_]+", " ", stem)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_source_meta(col, path):
    """Fill in source_meta for a new source: title (PDF metadata → first
    heading → prettified filename) and a plausible years range (regex over
    title + filename). Never overwrites user-edited values."""
    if path.name in col.source_meta and col.source_meta[path.name].get("title"):
        return
    title = None
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            t = (PdfReader(str(path)).metadata or {}).get("/Title") or ""
            t = re.sub(r"\s+", " ", str(t)).strip()
            if 4 <= len(t) <= 120 and not t.lower().startswith(("microsoft",
                                                                "untitled")):
                title = t
        except Exception:
            pass
    title = title or _prettify(path.stem)
    years = [int(y) for y in re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b",
                                        title + " " + path.name)]
    col.set_source_meta(path.name, title=title,
                        years=[min(years), max(years)] if years else None)


# ------------------------------------------------------ stage 2: embedding --

def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vectors_stale(out, src_file, n_items):
    """Content-based staleness: vectors must have been built from exactly
    this source file (hash), with the current model, one vector per item.
    Catches re-chunking, model switches, and partial writes — no mtimes."""
    if not out.exists():
        return True
    try:
        import numpy as np
        f = np.load(out)
        if "model" not in f.files or str(f["model"]) != config.EMBED_MODEL:
            return True
        if "src_hash" not in f.files or str(f["src_hash"]) != file_hash(src_file):
            return True
        return len(f["vectors"]) != n_items
    except Exception:
        return True


def embed_source(col, chunk_file, force=False, shared=None):
    """Stage 2 for one already-chunked source: embeds its chunks and — only
    if the Stage-1b summary artifact already exists as text — its summaries.
    Vector files carry the model name and a hash of the exact source file,
    so any future re-chunk or re-summarize triggers re-embedding on the
    next plain ingest run. Pass `shared` (a dict) across calls so the
    embedding model is loaded once per run, not once per source. Returns
    (status, shared)."""
    parts, todo = [], []
    shared = shared if shared is not None else {}

    out = col.vectors_dir / (chunk_file.stem + ".npz")
    chunks = [json.loads(l) for l in open(chunk_file)]
    if force or vectors_stale(out, chunk_file, len(chunks)):
        todo.append(([c["text"] for c in chunks], out, chunk_file,
                     f"{len(chunks)} chunks"))

    summ_file = col.summaries_dir / chunk_file.name
    if summ_file.exists():                      # summaries must pre-exist as text
        sout = col.vectors_dir / (chunk_file.stem + ".summaries.npz")
        summs = [json.loads(l) for l in open(summ_file)]
        if summs and (force or vectors_stale(sout, summ_file, len(summs))):
            todo.append(([f"{s['essay']}. {s['summary']} {s['keywords']}"
                          for s in summs], sout, summ_file,
                         f"{len(summs)} summaries"))

    if not todo:
        return "up to date", shared
    try:
        import numpy as np
        if "model" not in shared:
            from sentence_transformers import SentenceTransformer
            shared["model"] = SentenceTransformer(config.EMBED_MODEL)
    except ImportError:
        return ("skipped (sentence-transformers not installed — BM25 will "
                "be used)"), shared
    model = shared["model"]
    col.vectors_dir.mkdir(parents=True, exist_ok=True)
    for texts, out, src, label in todo:
        vecs = model.encode(texts, normalize_embeddings=True,
                            batch_size=config.EMBED_BATCH,
                            show_progress_bar=True)
        np.savez(out, vectors=np.asarray(vecs, dtype="float32"),
                 model=np.array(config.EMBED_MODEL),
                 src_hash=np.array(file_hash(src)))
        parts.append(label)
    return "embedded " + " + ".join(parts), shared


# --------------------------------------------------------------- pipeline --

def run_pipeline(col, force=False, summaries=True, progress=None, workers=None):
    """The whole pipeline for one collection. `progress` is an optional
    callable receiving {stage, source, status, i, total, …} events (the
    server streams them as NDJSON; the CLI prints them). `workers` limits
    the chunk-stage process pool (server passes 1 — threads inside a
    request must not fork)."""
    def emit(**ev):
        if progress:
            progress(ev)
        else:
            src = ev.get("source", "")
            print(f"[{ev['stage']}] {src}: {ev.get('status', '')}")

    col.ensure_dirs()
    sources = col.source_files()
    if not sources:
        emit(stage="error", status=f"no supported documents in "
             f"{col.sources_dir}")
        return {"chunks": 0, "sources": 0}

    manifest = load_manifest(col)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Stage 1 — chunk every new/changed source FIRST. Sources are
    # independent; a process pool when we may fork, else serial.
    import os
    if workers is None:
        workers = config.INGEST_WORKERS or max((os.cpu_count() or 2) // 2, 1)
    jobs = list(sources)

    def record(src, res):
        entry = manifest.setdefault(src.name, {})
        if res:
            _, n, rung = res
            entry.update({"chunked_at": now, "n_chunks": n,
                          "segmentation": rung})
            emit(stage="chunk", source=src.name, status=f"{n} chunks "
                 f"(segmentation: {rung})", n=n, total=len(jobs))
        else:
            emit(stage="chunk", source=src.name, status="up to date",
                 total=len(jobs))
        save_manifest(col, manifest)

    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = {pool.submit(chunk_source, col, src, force): src
                       for src in jobs}
            for fut in as_completed(futures):
                record(futures[fut], fut.result())
    else:
        for src in jobs:
            try:
                record(src, chunk_source(col, src, force))
            except Exception as e:
                emit(stage="chunk", source=src.name, status=f"FAILED: {e}")
                manifest.setdefault(src.name, {})["error"] = str(e)
                save_manifest(col, manifest)

    # metadata auto-detection for sources the user hasn't titled yet
    for src in sources:
        try:
            detect_source_meta(col, src)
        except Exception:
            pass

    # Stage 1c/1d — per-section dates + quotable-passage bank
    # (cheap regex sidecars; hash-idempotent)
    import dates
    dates.run_all(col, force)
    emit(stage="dates", status="done")
    import quotes_bank
    quotes_bank.run_all(col, force)
    emit(stage="quotes", status="done")

    # Stage 1b — summarize (generated like chunks, BEFORE any embedding);
    # reads only the completed chunk files written above
    if summaries:
        import summarize
        ok = summarize.run_all(col, force, manifest,
                               lambda m: save_manifest(col, m),
                               progress=progress)
        emit(stage="summarize", status="done" if ok else
             "skipped (LLM unreachable)")

    # Stage 2 — embed only sources whose chunking is complete
    shared = {}          # one embedding model for the whole run
    for src in sources:
        chunk_file = col.chunks_dir / (src.stem + ".jsonl")
        if not chunk_file.exists():
            emit(stage="embed", source=src.name, status="NOT chunked, skipping")
            continue
        status, shared = embed_source(col, chunk_file, force, shared)
        entry = manifest.setdefault(src.name, {})
        entry["embed_status"] = status
        entry["embedded_at"] = now if "embedded" in status else \
            entry.get("embedded_at")
        emit(stage="embed", source=src.name, status=status)
        save_manifest(col, manifest)

    total = sum(m.get("n_chunks", 0) for m in manifest.values())
    emit(stage="done", status=f"{len(sources)} source(s), {total} chunks")
    return {"chunks": total, "sources": len(sources)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "-c", required=True,
                    help="collection id (see: python serve.py, or "
                         "library.list_all())")
    ap.add_argument("--force", action="store_true", help="re-process everything")
    ap.add_argument("--no-summaries", action="store_true",
                    help="skip stage 1b (section summaries)")
    args = ap.parse_args()
    try:
        col = library.get(args.collection)
    except KeyError:
        ids = ", ".join(c.id for c in library.list_all()) or "(none yet)"
        sys.exit(f"Unknown collection {args.collection!r}. Available: {ids}")
    run_pipeline(col, force=args.force, summaries=not args.no_summaries)


if __name__ == "__main__":
    main()
