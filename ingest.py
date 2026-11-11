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
skipped (content-hash staleness everywhere: the manifest stores each
source's sha256 + chunk params for stage 1; vectors/summaries hash their
inputs). Derived data of sources that disappeared is pruned first.

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
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
import library


# ---------------------------------------------------------------- manifest --

MANIFEST_SCHEMA = 2
# manifest.json = {source_filename: entry, ...} plus reserved keys starting
# with "_" (e.g. "_schema"), which are never source names. Entries stay keyed
# by source FILENAME in every schema; 2 = written alongside the schema-2
# (full-filename) artifact layout, entries may carry "id_prefix".


def load_manifest_raw(col):
    """The manifest file as stored, reserved "_" keys included."""
    if col.manifest_file.exists():
        return json.loads(col.manifest_file.read_text())
    return {}


def load_manifest(col):
    """{source_filename: entry} — reserved "_" keys are stripped, so callers
    can iterate it as a pure per-source map."""
    return {k: v for k, v in load_manifest_raw(col).items()
            if not k.startswith("_")}


def save_manifest(col, m):
    """Atomic write; reserved keys are (re)added here, so a dict from
    load_manifest round-trips safely."""
    out = {"_schema": MANIFEST_SCHEMA}
    out.update({k: v for k, v in m.items() if not k.startswith("_")})
    library.write_json_atomic(col.manifest_file, out, indent=2)


def prune_removed(col, sources, manifest, emit):
    """Delete derived artifacts and manifest entries of sources that are no
    longer in the collection. Only files inside col.data_dir whose name
    parses (library.DERIVED_FILES) to an artifact key with no live source are
    touched — never the sources folder, conversations/, or eval.jsonl.
    Mutates `manifest`; emits one {stage: "prune"} event per pruned source.
    Returns the list of pruned names."""
    live = {s.name for s in sources}
    live_keys = {col.artifact_key(s.name) for s in sources}
    data_root = col.data_dir.resolve()
    gone = {}                                  # artifact key -> [paths]
    for p, keys in col.derived_files_on_disk().items():
        if keys & live_keys:
            continue
        if data_root not in p.resolve().parents:
            continue
        gone.setdefault(sorted(keys)[0], []).append(p)
    # manifest entries for vanished sources, matched to their artifact key
    stale_names = [n for n in manifest if n not in live]
    names_by_key = {}
    for n in stale_names:
        names_by_key.setdefault(col.artifact_key(n), []).append(n)
    pruned = []
    for key in sorted(set(gone) | set(names_by_key)):
        removed = 0
        for p in gone.get(key, []):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        for n in names_by_key.get(key, []):
            manifest.pop(n, None)
        label = ", ".join(names_by_key.get(key, [])) or key
        emit(stage="prune", source=label,
             status=f"source removed — deleted {removed} derived file(s)",
             removed=removed)
        pruned.append(label)
    return pruned


# ------------------------------------------------- schema-2 layout migration --

def _old_stem_key(name):
    """Schema-1 artifact key of a source: its stem. Used ONLY to recognise
    files written by the old layout during migration."""
    return Path(name).stem


def migrate_layout(col, sources=None, emit=None):
    """Re-key a schema-1 collection (derived files named after the source
    STEM) to the schema-2 layout (Collection.artifact_key = full filename)
    by RENAMING files — chunk/paras/dates/quotes/summary/vector bytes are
    untouched, so nothing is re-chunked, re-embedded or re-summarized:
    every staleness hash in those files (vectors' src_hash, summaries'
    .src sidecar, dates/quotes "_hash") is a hash of file CONTENT (chunk or
    summary file bytes), never of a file name, and the manifest is keyed by
    source filename, which does not change.

    Per old stem-keyed file:
      - exactly one current source has that stem → renamed to its new key
        (if the new-key file already exists, the new one wins and the old
        file is deleted);
      - several current sources share the stem → the file is ambiguous
        (whichever source wrote last overwrote the other) → deleted, so
        each source is rebuilt from scratch by the following stages;
      - no current source has it → left alone (prune_removed handles it).
    A file whose name already parses to a current schema-2 key is kept as
    is. Only files inside col.data_dir are touched — never the sources
    folder, conversations/ or eval.jsonl. Afterwards collection.json gets
    "schema": COLLECTION_SCHEMA (and the manifest "_schema" is bumped).
    Idempotent: a schema >= 2 collection returns at once.
    Returns {"renamed": n, "deleted": n} or None when nothing was to do."""
    if (col.meta.get("schema") or 1) >= 2:
        return None
    if not col.sources_dir.is_dir():
        return None            # unmounted linked folder: cannot attribute
    emit = emit or (lambda **ev: None)
    sources = col.source_files() if sources is None else sources
    by_stem = {}
    for s in sources:
        by_stem.setdefault(_old_stem_key(s.name), []).append(s.name)
    new_keys = {col.artifact_key(s.name) for s in sources}
    data_root = col.data_dir.resolve()
    renamed = deleted = 0
    ambiguous = set()
    for d in dict.fromkeys(d for d, _ in library.DERIVED_FILES):
        folder = getattr(col, d)
        if not folder.is_dir():
            continue
        suffixes = [sx for dd, sx in library.DERIVED_FILES if dd == d]
        for p in sorted(folder.iterdir()):
            if not p.is_file() or data_root not in p.resolve().parents:
                continue
            if any(p.name.endswith(sx) and p.name[:-len(sx)] in new_keys
                   for sx in suffixes):
                continue       # already schema-2 named
            # parse as the old writer named it: the first (longest) suffix
            # of this dir in DERIVED_FILES that matches
            suf = next((sx for sx in suffixes if p.name.endswith(sx)
                        and len(p.name) > len(sx)), None)
            if suf is None:
                continue
            old_key = p.name[:-len(suf)]
            owners = by_stem.get(old_key, [])
            if len(owners) == 1:
                target = folder / (col.artifact_key(owners[0]) + suf)
                if target.exists():
                    p.unlink()
                    deleted += 1
                else:
                    os.replace(p, target)
                    renamed += 1
            elif len(owners) > 1:
                p.unlink()
                deleted += 1
                ambiguous.add(old_key)
    col.meta["schema"] = library.COLLECTION_SCHEMA
    col.save()
    if col.manifest_file.exists():
        save_manifest(col, load_manifest(col))
    emit(stage="migrate", source="",
         status=f"re-keyed derived files to schema {library.COLLECTION_SCHEMA}:"
                f" {renamed} renamed, {deleted} deleted"
                + (f" (ambiguous stems: {', '.join(sorted(ambiguous))})"
                   if ambiguous else ""),
         renamed=renamed, deleted=deleted)
    return {"renamed": renamed, "deleted": deleted}


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


def _section_index(starts, page):
    """Index of the section containing `page` given ordered section start
    pages (same walk as essay_for_page), or -1 before the first section."""
    j = -1
    for k, start in enumerate(starts):
        if page >= start:
            j = k
        else:
            break
    return j


def _unique_titles(titles_seen, title):
    """Distinct display title for a repeated section title within one
    source: first 'X', then 'X (2)', 'X (3)', … (`titles_seen` is a set the
    caller keeps per source)."""
    if title not in titles_seen:
        titles_seen.add(title)
        return title
    n = 2
    while f"{title} ({n})" in titles_seen:
        n += 1
    titles_seen.add(f"{title} ({n})")
    return f"{title} ({n})"


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
            # keyed by TOC entry index, not title: two TOC entries with the
            # same title stay two sections (chunk_source suffixes " (2)")
            sections = {}
            for i in range(offset + 1, len(reader.pages)):
                printed = i - offset
                raw = reader.pages[i].extract_text() or ""
                if raw.strip():
                    j = _section_index([s for _, s in toc], printed)
                    essay = toc[j][0] if j >= 0 else "Front matter"
                    sections.setdefault(j, (essay, []))[1].append(
                        (printed, clean_page(raw, printed, title)))
            yield from sections.values()
        return "meta", gen_meta()

    bounds = _outline_sections(reader)
    rung = "outline" if bounds else None
    if not bounds:
        rung, bounds = _auto_sections(reader)

    if bounds:                            # sections found automatically

        def gen_auto():
            sections = {}
            offset = bounds[0][2]
            starts = [s for _, s, _ in bounds]
            for i in range(len(reader.pages)):
                raw = reader.pages[i].extract_text() or ""
                if not raw.strip():
                    continue
                # keyed by bound index, not title (duplicate titles stay
                # distinct); -1 = front matter before the first section
                j = _section_index(starts, i)
                title = bounds[j][0] if j >= 0 else work
                label = i - offset if (offset is not None and i > offset) else i + 1
                sections.setdefault(j, (title, []))[1].append(
                    (label, clean_page(raw, label, work)))
            yield from sections.values()
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


MIN_CHUNK_WORDS = 40


def chunk_params(col):
    """Effective chunking settings for a collection — stored in the manifest
    so a settings change re-chunks."""
    return {"chunk_words": col.setting("CHUNK_WORDS", config.CHUNK_WORDS),
            "overlap_words": col.setting("CHUNK_OVERLAP_WORDS",
                                         config.CHUNK_OVERLAP_WORDS)}


ID_PREFIX_LEN = 30


def chunk_id_prefixes(names):
    """{source filename: chunk-id prefix} for every source of a collection.

    Chunk ids are f"{prefix}-{section title[:40]}-{word offset}". The
    prefix is the source stem cut to ID_PREFIX_LEN — UNCHANGED from schema
    1, so ids (and saved conversations, quote banks, vectors) of existing
    collections stay valid — unless that cut stem is shared with another
    source of the collection (report.md + report.txt, or two long names
    equal in their first 30 characters). Then every source of the group
    gets the cut stem + its own suffix ("report.md", "report.txt"), and if
    that is still not unique (or clashes with another source's prefix),
    "~" + sha1(filename)[:8] on top. Pure function of the set of names."""
    names = sorted(set(names))
    base = {n: Path(n).stem[:ID_PREFIX_LEN] for n in names}
    counts = {}
    for b in base.values():
        counts[b] = counts.get(b, 0) + 1
    out = {n: (base[n] if counts[base[n]] == 1
               else base[n] + Path(n).suffix) for n in names}
    taken = {}
    for n in names:
        taken.setdefault(out[n], []).append(n)
    for p, group in taken.items():
        if len(group) > 1:
            for n in group:            # a plain (non-colliding) prefix
                if out[n] != base[n]:      # is never changed
                    out[n] = f"{p}~" + hashlib.sha1(
                        n.encode("utf-8")).hexdigest()[:8]
    return out


def chunk_id_prefix(col, path):
    """Chunk-id prefix of one source given the collection's current
    sources (see chunk_id_prefixes)."""
    names = {s.name for s in col.source_files()} | {path.name}
    return chunk_id_prefixes(names)[path.name]


def chunk_is_current(col, path, prev, src_sha256=None, params=None,
                     id_prefix=None):
    """Content-hash staleness for stage 1: chunk + paras files exist and the
    manifest entry `prev` records this exact source sha256, the current
    chunk params and the current chunk-id prefix (entries written before
    the prefix was recorded used the cut stem). mtimes are never consulted."""
    if not (col.chunk_file(path.name).exists()
            and (col.paras_dir / (col.artifact_key(path.name)
                                  + ".json")).exists()):
        return False
    if not prev:
        return False
    sha = src_sha256 or file_hash(path)
    if id_prefix is None:
        id_prefix = chunk_id_prefix(col, path)
    return (prev.get("src_sha256") == sha
            and prev.get("chunk_params") == (params or chunk_params(col))
            and prev.get("id_prefix", path.stem[:ID_PREFIX_LEN]) == id_prefix)


def chunk_source(col, path, force=False, *, prev=None, src_sha256=None,
                 id_prefix=None):
    """Stage 1 for one source. Also emits a paragraph-map sidecar
    (data/paras/<key>.json: chunk_id -> in-chunk word offsets starting a
    paragraph) used only for display; chunk bytes are unaffected.

    Skipped (returns None) only when not `force` and `prev` — the source's
    manifest entry, passed in because the manifest is owned by the parent
    process — records the same source sha256 and chunk params
    (see chunk_is_current). Without `prev` it always chunks. `id_prefix`
    (chunk_id_prefixes) is computed from the collection's sources when not
    given. Returns (chunk_file, n_chunks, segmentation_rung) or None if
    skipped."""
    key = col.artifact_key(path.name)
    out = col.chunks_dir / (key + ".jsonl")
    para_out = col.paras_dir / (key + ".json")
    if id_prefix is None:
        id_prefix = chunk_id_prefix(col, path)
    if not force and chunk_is_current(col, path, prev, src_sha256,
                                      id_prefix=id_prefix):
        return None
    rung, sections = extract_sections(col, path)

    col.chunks_dir.mkdir(parents=True, exist_ok=True)
    col.paras_dir.mkdir(parents=True, exist_ok=True)
    n, para_map = 0, {}
    params = chunk_params(col)
    chunk_w, overlap_w = params["chunk_words"], params["overlap_words"]
    seen_titles = set()
    with open(out, "w") as f:
        for raw_title, pages in sections:
            # repeated titles within a source stay distinct: 'X', 'X (2)', …
            title = _unique_titles(seen_titles, raw_title)
            tid = raw_title[:40] + title[len(raw_title):]
            words, word_pages, paras = words_with_paras(pages)
            prev_end = 0               # end offset of the last kept window
            for start, ck in chunk_words(words, chunk_w, overlap_w):
                if not ck:
                    continue
                # a short window is kept unless every word in it is already
                # in the previous chunk: the first window of a section is
                # always kept, and no word of a section is ever lost
                if (len(ck) < MIN_CHUNK_WORDS and start > 0
                        and start + len(ck) <= prev_end):
                    continue
                prev_end = start + len(ck)
                cid = f"{id_prefix}-{tid}-{start}"
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


def vectors_stale(out, src_file, n_items, tag=None):
    """Content-based staleness: vectors must have been built from exactly
    this source file (hash), with the current model, pooling and max
    sequence length (encoder.current_tag(), or `tag`), one vector per item.
    Catches re-chunking, model/pooling switches, and partial writes — no
    mtimes. Vectors without pooling/max_len tags (written before they
    existed) are stale whenever an encoder is available to compare with."""
    if not out.exists():
        return True
    try:
        import numpy as np
        import encoder
        f = np.load(out)
        if encoder.tag_mismatch(f, tag):
            return True
        if "src_hash" not in f.files or str(f["src_hash"]) != file_hash(src_file):
            return True
        return len(f["vectors"]) != n_items
    except Exception:
        return True


def embed_source(col, chunk_file, force=False, shared=None, progress=None):
    """Stage 2 for one already-chunked source: embeds its chunks and — only
    if the Stage-1b summary artifact already exists as text — its summaries,
    with the process-wide encoder.get_encoder() (ONNX when bundled, else
    sentence-transformers) — the same instance rag uses for queries.
    Vector files carry the model name, pooling, max length and a hash of
    the exact source file, so any future re-chunk, re-summarize or encoder
    change triggers re-embedding on the next plain ingest run.
    `progress(done, total)` is called after every encode batch (total =
    chunks + summaries to embed for this source). `shared` is kept for
    callers that pass one dict across calls; it caches the encoder under
    "model". Returns (status, shared)."""
    parts, todo = [], []
    shared = shared if shared is not None else {}

    key = col.key_of_chunk_file(chunk_file)
    out = col.vectors_dir / (key + ".npz")
    chunks = [json.loads(l) for l in open(chunk_file) if l.strip()]
    if force or vectors_stale(out, chunk_file, len(chunks)):
        todo.append(([c["text"] for c in chunks], out, chunk_file,
                     f"{len(chunks)} chunks"))

    summ_file = col.summaries_dir / (key + ".jsonl")
    if summ_file.exists():                      # summaries must pre-exist as text
        sout = col.vectors_dir / (key + ".summaries.npz")
        summs = [json.loads(l) for l in open(summ_file) if l.strip()]
        if summs and (force or vectors_stale(sout, summ_file, len(summs))):
            todo.append(([f"{s['essay']}. {s['summary']} {s['keywords']}"
                          for s in summs], sout, summ_file,
                         f"{len(summs)} summaries"))

    if not todo:
        return "up to date", shared
    try:
        import numpy as np
        import encoder
        if "model" not in shared:
            shared["model"] = encoder.get_encoder()
    except ImportError:
        return ("skipped (no encoder: neither models/encoder-onnx nor "
                "sentence-transformers — BM25 will be used)"), shared
    model = shared["model"]
    tag = model.tag
    total = sum(len(t) for t, *_ in todo)
    base = 0
    col.vectors_dir.mkdir(parents=True, exist_ok=True)
    for texts, out, src, label in todo:
        def batch_done(done, _n, base=base):
            if progress:
                progress(base + done, total)
        vecs = model.encode(texts, normalize_embeddings=True,
                            batch_size=config.EMBED_BATCH,
                            progress=batch_done)
        base += len(texts)
        # write-then-rename: a crash never leaves a half-written vector file
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez(tmp, vectors=np.asarray(vecs, dtype="float32"),
                 model=np.array(tag["model"]),
                 pooling=np.array(tag["pooling"]),
                 max_len=np.array(int(tag["max_len"])),
                 src_hash=np.array(file_hash(src)))
        os.replace(tmp, out)
        parts.append(label)
    return "embedded " + " + ".join(parts), shared


# --------------------------------------------------------------- pipeline --

def record_chunked(entry, res, now, src_sha256, params, id_prefix):
    """Update one manifest entry after chunk_source returned `res`, so the
    next run sees the source as current (chunk_is_current)."""
    _, n, rung = res
    entry.update({"chunked_at": now, "n_chunks": n, "segmentation": rung,
                  "src_sha256": src_sha256, "chunk_params": params,
                  "id_prefix": id_prefix})
    entry.pop("error", None)


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
    if not col.sources_dir.is_dir():
        # e.g. a linked folder on an unmounted drive: never prune on this
        emit(stage="error", status=f"source folder not found: "
             f"{col.sources_dir}")
        return {"chunks": 0, "sources": 0}
    sources = col.source_files()
    # schema-1 (stem-keyed) layout → rename to full-filename keys BEFORE the
    # prune, which would otherwise see the old names as orphans
    migrate_layout(col, sources, emit)
    manifest = load_manifest(col)

    # Prune derived data of sources that no longer exist, BEFORE any stage
    # (dates/quotes/summaries/retrieval all glob the chunks dir)
    if prune_removed(col, sources, manifest, emit):
        save_manifest(col, manifest)

    if not sources:
        emit(stage="error", status=f"no supported documents in "
             f"{col.sources_dir}")
        return {"chunks": 0, "sources": 0}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Stage 1 — chunk every new/changed source FIRST. Sources are
    # independent; a process pool when we may fork, else serial.
    # Staleness = source sha256 + chunk params + chunk-id prefix, recorded
    # in the manifest.
    if workers is None:
        workers = config.INGEST_WORKERS or max((os.cpu_count() or 2) // 2, 1)
    jobs = list(sources)
    params = chunk_params(col)
    shas = {src.name: file_hash(src) for src in jobs}
    prefixes = chunk_id_prefixes(src.name for src in jobs)

    def job_kw(src):
        return {"prev": manifest.get(src.name), "src_sha256": shas[src.name],
                "id_prefix": prefixes[src.name]}

    def record(src, res):
        entry = manifest.setdefault(src.name, {})
        if res:
            _, n, rung = res
            record_chunked(entry, res, now, shas[src.name], params,
                           prefixes[src.name])
            emit(stage="chunk", source=src.name, status=f"{n} chunks "
                 f"(segmentation: {rung})", n=n, total=len(jobs))
        else:
            emit(stage="chunk", source=src.name, status="up to date",
                 total=len(jobs))
        save_manifest(col, manifest)

    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = {pool.submit(chunk_source, col, src, force,
                                   **job_kw(src)): src
                       for src in jobs}
            for fut in as_completed(futures):
                record(futures[fut], fut.result())
    else:
        for src in jobs:
            try:
                record(src, chunk_source(col, src, force, **job_kw(src)))
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
        chunk_file = col.chunk_file(src.name)
        if not chunk_file.exists():
            emit(stage="embed", source=src.name, status="NOT chunked, skipping")
            continue
        def embed_progress(done, n, name=src.name):
            emit(stage="embed", source=name, done=done, total=n,
                 status=f"embedding {done}/{n}")
        status, shared = embed_source(col, chunk_file, force, shared,
                                      progress=embed_progress)
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
