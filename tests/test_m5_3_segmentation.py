"""M5-3 (P2-3): heading segmentation for PDF (font size/weight), HTML
(<h1>/<h2>), TXT (setext / ALL-CAPS / numbered / '#') and DOCX (Heading 2).
Small fixture documents are generated here, not added to CLEAN_FILES."""
import json
import textwrap
from collections import Counter

import pytest

import ingest
import library
import rag

LOREM = ("The survey team walked the eastern ridge each morning and wrote "
         "down every change they saw in the soil, the plants and the small "
         "streams that cross the old track. Their notes describe slow "
         "erosion, new growth after the fire, and a steady return of birds "
         "to the lower slopes over three seasons of careful field work.")


def _col(tmp_path, files):
    d = tmp_path / "docs"
    d.mkdir(exist_ok=True)
    for name, data in files.items():
        p = d / name
        if callable(data):
            data(p)
        elif isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data)
    col = library.create("Seg", link=str(d))
    ingest.run_pipeline(col, summaries=False, workers=1,
                        progress=lambda e: None)
    return library.get(col.id)


def _rung(col, name):
    return ingest.load_manifest(col)[name]["segmentation"]


def _sections(col, name):
    out = []
    for c in rag.load_chunks(col):
        if c["source"] == name and c["essay"] not in out:
            out.append(c["essay"])
    return out


def _words(col, name):
    return Counter(w for c in rag.load_chunks(col) if c["source"] == name
                   for w in c["text"].split())


# ------------------------------------------------------------------- TXT --

TXT = f"""Field Report
============

{LOREM}

GEOLOGY OF THE RIDGE

{LOREM}

2. Plant Recovery

{LOREM}
"""


def test_txt_headings(tmp_path):
    col = _col(tmp_path, {"field.txt": TXT})
    assert _rung(col, "field.txt") == "headings"
    assert _sections(col, "field.txt") == [
        "Field Report", "GEOLOGY OF THE RIDGE", "2. Plant Recovery"]
    w = _words(col, "field.txt")
    assert "====" not in "".join(w)                    # underline dropped
    assert w["GEOLOGY"] == 0                           # heading not in body
    assert w["survey"] == 3                            # no body word lost


def test_txt_short_list_items_are_not_headings(tmp_path):
    txt = ("Bicycle care notes.\n\n1. Clean the chain\n\nWipe it.\n\n"
           "2. Check the tyres\n\nPump them.\n\n" + LOREM + "\n")
    col = _col(tmp_path, {"care.txt": txt})
    assert _rung(col, "care.txt") == "none"
    assert _sections(col, "care.txt") == ["care"]


def test_txt_single_heading_stays_none(tmp_path):
    col = _col(tmp_path, {"one.txt": f"INTRODUCTION\n\n{LOREM}\n"})
    assert _rung(col, "one.txt") == "none"


def test_txt_heading_mid_paragraph_ignored(tmp_path):
    # an ALL-CAPS line inside a paragraph (no blank line before) is text
    txt = f"{LOREM}\nNASA AND ESA\n{LOREM}\n\nRESULTS\n\n{LOREM}\n"
    col = _col(tmp_path, {"mid.txt": txt})
    assert "NASA AND ESA" not in _sections(col, "mid.txt")


# ------------------------------------------------------------------ HTML --

def test_html_h1_h2_sections(tmp_path):
    html = ("<html><head><title>Ridge</title><script>var x=1;</script>"
            f"</head><body><p>Intro words here.</p><h1>Geology</h1><p>{LOREM}"
            f"</p><h2>Soils <em>and</em> Rocks</h2><p>{LOREM}</p>"
            f"<h3>Detail</h3><p>Small part.</p><h2>Birds</h2><p>{LOREM}</p>"
            "</body></html>")
    col = _col(tmp_path, {"ridge.html": html})
    assert _rung(col, "ridge.html") == "headings"
    assert _sections(col, "ridge.html") == [
        "Ridge", "Geology", "Soils and Rocks", "Birds"]
    soils = [c for c in rag.load_chunks(col) if c["essay"] == "Soils and Rocks"]
    assert "Detail" in soils[0]["text"] and "Small part." in soils[0]["text"]
    assert all("var x" not in c["text"] for c in rag.load_chunks(col))


def test_html_single_h1_unchanged(tmp_path):
    html = (f"<html><head><title>T</title></head><body><h1>Only</h1>"
            f"<p>{LOREM}</p></body></html>")
    col = _col(tmp_path, {"one.html": html})
    assert _rung(col, "one.html") == "none"
    assert _sections(col, "one.html") == ["T"]
    assert rag.load_chunks(col)[0]["text"].startswith("Only ")


# ------------------------------------------------------------------ DOCX --

def test_docx_heading_2(tmp_path):
    docx = pytest.importorskip("docx")

    def make(p):
        d = docx.Document()
        d.add_heading("Survey", level=1)
        d.add_paragraph(LOREM)
        d.add_heading("Soils", level=2)
        d.add_paragraph(LOREM)
        d.add_heading("Minor note", level=3)
        d.add_paragraph("Inside soils.")
        d.save(str(p))
    col = _col(tmp_path, {"survey.docx": make})
    assert _rung(col, "survey.docx") == "headings"
    assert _sections(col, "survey.docx") == ["Survey", "Soils"]


# ------------------------------------------------------------------- PDF --

def _esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf(pages):
    """pages: [[(font 'F1'|'F2', size, text), …]] one line each, top down.
    F1 Helvetica (body), F2 Helvetica-Bold. No outline, no TOC."""
    objs = []

    def add(b):
        objs.append(b)
        return len(objs)
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    pages_id = len(objs) + 1 + 2 * len(pages)
    ids = []
    for lines in pages:
        ops, y = [], 740
        for font, size, text in lines:
            ops.append(f"BT /{font} {size} Tf 72 {y} Td ({_esc(text)}) Tj ET")
            y -= size + 6
        st = "\n".join(ops).encode("latin-1")
        c = add(b"<< /Length %d >>\nstream\n" % len(st) + st + b"\nendstream")
        ids.append(add((f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox "
                        f"[0 0 612 792] /Resources << /Font << /F1 {f1} 0 R "
                        f"/F2 {f2} 0 R >> >> /Contents {c} 0 R >>").encode()))
    kids = " ".join(f"{i} 0 R" for i in ids)
    assert add(f"<< /Type /Pages /Kids [{kids}] /Count {len(ids)} >>"
               .encode()) == pages_id
    cat = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    buf, offs = bytearray(b"%PDF-1.4\n"), []
    for i, b in enumerate(objs, start=1):
        offs.append(len(buf))
        buf += b"%d 0 obj\n" % i + b + b"\nendobj\n"
    x = len(buf)
    buf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        buf += b"%010d 00000 n \n" % o
    buf += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, cat, x))
    return bytes(buf)


def _body(text):
    return [("F1", 11, ln) for ln in textwrap.wrap(text, 80)]


HEADER = ("F2", 11, "Ridge Survey Notes")     # running header, every page
PDF_PAGES = [
    [HEADER, ("F1", 11, "Prepared by the field team"),
     ("F2", 20, "Geology of the Ridge")] + _body(LOREM),
    [HEADER] + _body(LOREM) + [("F2", 11, "Plant Recovery")] + _body(LOREM),
    [HEADER, ("F1", 18, "Birds Return")] + _body(LOREM)
    + [("F2", 11, "Bold sentence that ends with a period.")] + _body(LOREM),
]


def test_pdf_font_headings(tmp_path):
    col = _col(tmp_path, {"ridge.pdf": _pdf(PDF_PAGES)})
    assert _rung(col, "ridge.pdf") == "fonts"
    secs = _sections(col, "ridge.pdf")
    # front matter keeps the work title; the repeated bold header and the
    # bold sentence are not headings
    assert secs[1:] == ["Geology of the Ridge", "Plant Recovery",
                        "Birds Return"]
    assert "Ridge Survey Notes" not in secs
    w = _words(col, "ridge.pdf")
    assert w["survey"] == 5 and w["Survey"] == 3       # header stays as text
    assert w["Geology"] == 0 and w["Recovery"] == 0    # headings left body
    pages = {c["essay"]: (c["page_start"], c["page_end"])
             for c in rag.load_chunks(col) if c["source"] == "ridge.pdf"}
    # page 3's running header precedes "Birds Return", so it is (body)
    # text of "Plant Recovery"
    assert pages["Plant Recovery"] == (2, 3) and pages["Birds Return"][0] == 3


def test_pdf_without_headings_stays_none(tmp_path):
    flat = [_body(LOREM), _body(LOREM)]
    col = _col(tmp_path, {"flat.pdf": _pdf(flat)})
    assert _rung(col, "flat.pdf") == "none"


# ------------------------------------------------------------- staleness --

def test_old_segmenter_rechunks_once(tmp_path):
    col = _col(tmp_path, {"field.txt": TXT})
    m = ingest.load_manifest(col)
    assert m["field.txt"]["segmenter"] == ingest.SEGMENTER_VERSION
    src = col.source_path("field.txt")
    assert ingest.chunk_is_current(col, src, m["field.txt"])
    old = dict(m["field.txt"])
    old.pop("segmenter")                         # written before P2-3
    assert not ingest.chunk_is_current(col, src, old)
    m["field.txt"] = old
    ingest.save_manifest(col, m)
    ev = []
    ingest.run_pipeline(library.get(col.id), summaries=False, workers=1,
                        progress=ev.append)
    assert any(e.get("stage") == "chunk" and "chunks" in str(e.get("status"))
               for e in ev)
    assert ingest.load_manifest(col)["field.txt"]["segmenter"] == \
        ingest.SEGMENTER_VERSION


def test_split_on_headings_loses_no_line():
    lines = [(1, "A"), (1, "x " * 50), (2, "B"), (2, "C"), (2, "y " * 50),
             (3, "D"), (3, "short")]
    out = ingest._split_on_headings(lines, {0, 2, 3, 5}, "Doc")
    # B+C merge into one title; D has too few words → back into the body
    assert [t for t, _ in out] == ["A", "B C"]
    text = "\n".join(t for _, pages in out for _, t in pages)
    assert "D" in text.split() and "short" in text
    assert [p for p, _ in out[1][1]] == [2, 3]
    assert ingest._split_on_headings(lines, {0}, "Doc") is None
    assert json.dumps(out)
