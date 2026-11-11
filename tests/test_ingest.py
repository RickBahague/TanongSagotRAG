"""Stage 1 ingestion of every supported format + manifest + metadata."""
import ingest
import rag


def _sections(col, source):
    return sorted({c["essay"] for c in rag.load_chunks(col)
                   if c["source"] == source})


def test_every_format_ingests(make_collection):
    col = make_collection()
    m = ingest.load_manifest(col)
    rungs = {k: v["segmentation"] for k, v in m.items()}
    assert rungs == {"astronomy.pdf": "outline", "beekeeping.md": "headings",
                     "glaciers.docx": "headings", "lighthouses.html": "none",
                     "volcanoes.txt": "none"}
    assert not any("error" in v for v in m.values())
    assert col.ingest_events[-1]["stage"] == "done"


def test_section_titles(make_collection):
    col = make_collection()
    assert _sections(col, "beekeeping.md") == \
        ["Hive Structure", "Honey Harvest", "Swarming"]
    assert _sections(col, "glaciers.docx") == \
        ["Glacial Landforms", "Glacier Formation", "Ice Cores"]
    # html: the <title> names the section; script/style text is dropped
    assert _sections(col, "lighthouses.html") == \
        ["A Short History of Lighthouses"]
    html = [c for c in rag.load_chunks(col) if c["source"] == "lighthouses.html"]
    assert "var x" not in html[0]["text"] and "margin" not in html[0]["text"]
    assert {"Comets", "Nebulae", "Telescopes"} <= set(
        _sections(col, "astronomy.pdf"))


def test_pdf_pages_and_metadata_title(make_collection):
    col = make_collection(files=("astronomy.pdf",))
    by_essay = {c["essay"]: c for c in rag.load_chunks(col)}
    assert by_essay["Comets"]["page_start"] == 2
    assert by_essay["Nebulae"]["page_start"] == 4
    assert col.work_title("astronomy.pdf") == "Notes on Astronomy"


def test_chunk_text_is_heading_free_but_title_indexed(make_collection):
    col = make_collection(files=("beekeeping.md",))
    chunks = rag.load_chunks(col)
    assert not any(c["text"].startswith("#") for c in chunks)
    idx = rag.BM25Index(chunks)
    top = idx.search("swarming", 1)[0][1]
    assert top["essay"] == "Swarming"


def test_reingest_is_incremental(make_collection):
    from helpers import reingest
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    events = reingest(col)
    chunk_ev = [e for e in events if e["stage"] == "chunk"]
    assert chunk_ev and all(e["status"] == "up to date" for e in chunk_ev)


def test_sidecars_written(make_collection):
    col = make_collection(files=("beekeeping.md",))
    for d in (col.chunks_dir, col.paras_dir, col.dates_dir, col.quotes_dir):
        assert any(d.iterdir()), d


def test_managed_collection_ingests(make_collection):
    col = make_collection(files=("volcanoes.txt",), linked=False)
    assert {c["source"] for c in rag.load_chunks(col)} == {"volcanoes.txt"}


def test_chunking_respects_collection_settings(make_collection):
    col = make_collection(files=("volcanoes.txt",),
                          settings={"CHUNK_WORDS": 50,
                                    "CHUNK_OVERLAP_WORDS": 10})
    chunks = rag.load_chunks(col)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 50 for c in chunks)
