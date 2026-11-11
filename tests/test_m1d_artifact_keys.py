"""M1-D (P0-5): full-filename artifact keys, collection-unique chunk ids,
and the schema-1 (stem-keyed) → schema-2 migration by renaming."""
import json
import shutil
from pathlib import Path

import config
import ingest
import library
import rag
from helpers import CORPUS, reingest


# ------------------------------------------------------------- helpers --

def _to_stem_layout(col, names):
    """Rewrite col's data dir into the OLD layout: every derived file of
    each source in `names` renamed from its schema-2 key to the source
    stem; collection.json schema 1, manifest _schema 1, no id_prefix."""
    for name in names:
        new, old = col.artifact_key(name), Path(name).stem
        for d, suf in library.DERIVED_FILES:
            p = getattr(col, d) / (new + suf)
            if p.exists():
                p.rename(getattr(col, d) / (old + suf))
    meta = json.loads((col.root / "collection.json").read_text())
    meta["schema"] = 1
    (col.root / "collection.json").write_text(json.dumps(meta))
    raw = json.loads(col.manifest_file.read_text())
    raw["_schema"] = 1
    for k, v in raw.items():
        if not k.startswith("_"):
            v.pop("id_prefix", None)
    col.manifest_file.write_text(json.dumps(raw))
    return library.get(col.id)


def _fake_summary_and_vectors(col, key):
    """Summaries text + .src sidecar and chunk/summary vectors for one
    chunk file, hashed exactly as summarize.py / ingest.embed_source do."""
    import numpy as np
    cf = col.chunks_dir / (key + ".jsonl")
    n = sum(1 for l in open(cf) if l.strip())
    summ = col.summaries_dir / (key + ".jsonl")
    summ.write_text(json.dumps({"essay": "E", "source": "x", "summary": "s",
                                "keywords": "k", "page_start": 1,
                                "page_end": 1}) + "\n")
    (col.summaries_dir / (key + ".jsonl.src")).write_text(
        ingest.file_hash(cf))
    for out, src, k in ((col.vectors_dir / (key + ".npz"), cf, n),
                        (col.vectors_dir / (key + ".summaries.npz"), summ, 1)):
        np.savez(out, vectors=np.zeros((k, 4), dtype="float32"),
                 model=np.array(config.EMBED_MODEL),
                 pooling=np.array("cls"), max_len=np.array(512),
                 src_hash=np.array(ingest.file_hash(src)))


def _stage(events, stage):
    return [e for e in events if e["stage"] == stage]


# --------------------------------------------------------------- keys --

def test_artifact_key_is_full_filename():
    k = library.artifact_key_for
    assert k("report.md") == "report.md"
    assert k("report.txt") == "report.txt"
    assert k("Annual Report (2020).pdf") == "Annual Report (2020).pdf"
    # unsafe characters / over-long names: sanitized + hash, still unique
    a, b = k('a:b.pdf'), k('a?b.pdf')
    assert a != b and ":" not in a and "?" not in b
    long1, long2 = "x" * 300 + "1.pdf", "x" * 300 + "2.pdf"
    assert k(long1) != k(long2)
    assert len((k(long1) + ".summaries.npz").encode()) < 255
    assert k("report.md") == k("/some/dir/report.md")        # name only


def test_derived_paths_use_full_filename(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    assert (col.chunks_dir / "volcanoes.txt.jsonl").exists()
    assert (col.paras_dir / "volcanoes.txt.json").exists()
    assert (col.dates_dir / "volcanoes.txt.json").exists()
    assert (col.quotes_dir / "volcanoes.txt.jsonl").exists()
    on_disk = col.derived_files_on_disk()
    assert on_disk and set(on_disk) <= set(col.derived_paths("volcanoes.txt"))
    assert all("volcanoes.txt" in keys for keys in on_disk.values())
    assert json.loads((col.root / "collection.json").read_text())[
        "schema"] == library.COLLECTION_SCHEMA == 2


# ----------------------------------------------------------- chunk ids --

def test_chunk_id_prefix_rule():
    p = ingest.chunk_id_prefixes(["report.md", "report.txt", "volcanoes.txt",
                                  "report.md.pdf"])
    assert p["volcanoes.txt"] == "volcanoes"            # unchanged
    assert p["report.md.pdf"] == "report.md"            # plain stays plain
    assert p["report.txt"] == "report.txt"
    assert p["report.md"].startswith("report.md~")      # clash → + hash
    assert len(set(p.values())) == 4
    long = ["L" * 40 + " a.pdf", "L" * 40 + " b.pdf"]
    q = ingest.chunk_id_prefixes(long)
    assert len(set(q.values())) == 2


def test_same_stem_sources_distinct_ids(make_collection):
    col = make_collection(files=("report.md", "report.txt", "volcanoes.txt"))
    chunks = rag.load_chunks(col)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c["source"] for c in chunks} == {"report.md", "report.txt",
                                            "volcanoes.txt"}
    by_src = {}
    for c in chunks:
        by_src.setdefault(c["source"], set()).add(c["id"].split("-")[0])
    assert by_src["report.md"] == {"report.md"}
    assert by_src["report.txt"] == {"report.txt"}
    assert by_src["volcanoes.txt"] == {"volcanoes"}     # ids unchanged
    # files on disk carry raw unique ids too (not only after read-time dedup)
    raw = [json.loads(l)["id"] for f in col.chunks_dir.glob("*.jsonl")
           for l in open(f) if l.strip()]
    assert len(raw) == len(set(raw))
    # reader works for both same-stem sources
    rt = rag.Runtime(col)
    for c in chunks:
        if c["source"].startswith("report"):
            assert rt.passage(c["id"])["cited"]


def test_adding_same_stem_source_rechunks_sibling(make_collection):
    col = make_collection(files=("report.md",))
    assert all(c["id"].startswith("report-") for c in rag.load_chunks(col))
    shutil.copy2(CORPUS / "report.txt", col.sources_dir / "report.txt")
    ev = {e["source"]: e["status"] for e in _stage(reingest(col), "chunk")}
    assert ev["report.md"] != "up to date"            # its prefix changed
    ids = [c["id"] for c in rag.load_chunks(library.get(col.id))]
    assert len(ids) == len(set(ids))
    m = ingest.load_manifest(col)
    assert m["report.md"]["id_prefix"] == "report.md"
    # and stable afterwards
    assert all(e["status"] == "up to date"
               for e in _stage(reingest(col), "chunk"))


# ------------------------------------------------------------ migration --

def test_stem_keyed_collection_migrates(make_collection):
    files = ("volcanoes.txt", "beekeeping.md", "astronomy.pdf")
    col = make_collection(files=files)
    for name in files:
        _fake_summary_and_vectors(col, col.artifact_key(name))
    before_ids = sorted(c["id"] for c in rag.load_chunks(col))
    new_bytes = {p.relative_to(col.data_dir): p.read_bytes()
                 for p in col.derived_files_on_disk()}
    src_snapshot = {p.name: p.read_bytes() for p in col.sources_dir.iterdir()}
    col = _to_stem_layout(col, files)
    assert (col.chunks_dir / "volcanoes.jsonl").exists()
    assert not (col.chunks_dir / "volcanoes.txt.jsonl").exists()
    chunked_at = {k: v["chunked_at"]
                  for k, v in ingest.load_manifest(col).items()}

    events = reingest(col)

    mig = _stage(events, "migrate")
    assert len(mig) == 1 and mig[0]["deleted"] == 0
    assert mig[0]["renamed"] == len(new_bytes)
    assert not _stage(events, "prune")
    # renamed, byte-identical, nothing stem-keyed left
    after = {p.relative_to(col.data_dir): p.read_bytes()
             for p in col.derived_files_on_disk()}
    assert after == new_bytes
    assert not (col.chunks_dir / "volcanoes.jsonl").exists()
    # no re-chunk, and the content hashes in sidecars are still valid
    assert all(e["status"] == "up to date" for e in _stage(events, "chunk"))
    m = ingest.load_manifest(col)
    assert {k: v["chunked_at"] for k, v in m.items()} == chunked_at
    assert all(e["status"] == "up to date" for e in _stage(events, "embed"))
    key = col.artifact_key("volcanoes.txt")
    cf = col.chunks_dir / (key + ".jsonl")
    assert not ingest.vectors_stale(col.vectors_dir / (key + ".npz"), cf,
                                    m["volcanoes.txt"]["n_chunks"])
    assert (col.summaries_dir / (key + ".jsonl.src")).read_text() == \
        ingest.file_hash(cf)
    # chunks load with the same ids; reader + quote bank work
    col = library.get(col.id)
    assert sorted(c["id"] for c in rag.load_chunks(col)) == before_ids
    rt = rag.Runtime(col)
    cid = next(c["id"] for c in rt.index.chunks
               if c["source"] == "volcanoes.txt")
    assert rt.passage(cid)["cited"]
    assert rt.quote_bank and rt.random_quote() is not None
    assert {e["chunk_id"] for e in rt.quote_bank} <= set(rt.by_id)
    # schema bumped; sources untouched
    assert json.loads((col.root / "collection.json").read_text())[
        "schema"] == 2
    assert json.loads(col.manifest_file.read_text())["_schema"] == 2
    assert {p.name: p.read_bytes()
            for p in col.sources_dir.iterdir()} == src_snapshot


def test_migration_without_schema_key(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    col = _to_stem_layout(col, ("volcanoes.txt",))
    meta = json.loads((col.root / "collection.json").read_text())
    del meta["schema"]
    (col.root / "collection.json").write_text(json.dumps(meta))
    events = reingest(col)
    assert _stage(events, "migrate")[0]["renamed"] >= 4
    assert (col.chunks_dir / "volcanoes.txt.jsonl").exists()
    assert all(e["status"] == "up to date" for e in _stage(events, "chunk"))


def test_ambiguous_stem_migration_rebuilds_both(make_collection):
    col = make_collection(files=("report.md", "report.txt"))
    # old layout: both sources wrote report.*; the .txt overwrote the .md
    for d, suf in library.DERIVED_FILES:
        folder = getattr(col, d)
        (folder / ("report.md" + suf)).unlink(missing_ok=True)
        p = folder / ("report.txt" + suf)
        if p.exists():
            p.rename(folder / ("report" + suf))
    col = _to_stem_layout(col, ())
    assert (col.chunks_dir / "report.jsonl").exists()

    events = reingest(col)

    mig = _stage(events, "migrate")[0]
    assert mig["renamed"] == 0 and mig["deleted"] >= 4
    assert not list(col.data_dir.glob("*/report.json*"))
    ev = {e["source"]: e["status"] for e in _stage(events, "chunk")}
    assert ev["report.md"] != "up to date" and ev["report.txt"] != "up to date"
    chunks = rag.load_chunks(library.get(col.id))
    text = " ".join(c["text"] for c in chunks)
    assert "sourdough" in text and "derailleur" in text
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert (col.sources_dir / "report.md").exists()
    assert (col.sources_dir / "report.txt").exists()


def test_migration_idempotent(make_collection):
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    col = _to_stem_layout(col, ("volcanoes.txt", "beekeeping.md"))
    reingest(col)
    snap = {p: p.read_bytes() for p in col.derived_files_on_disk()}
    manifest = col.manifest_file.read_text()
    events = reingest(col)
    assert not _stage(events, "migrate") and not _stage(events, "prune")
    assert all(e["status"] == "up to date" for e in _stage(events, "chunk"))
    assert {p: p.read_bytes() for p in col.derived_files_on_disk()} == snap
    assert col.manifest_file.read_text() == manifest
    # calling it directly on a schema-2 collection is a no-op
    assert ingest.migrate_layout(library.get(col.id)) is None


def test_import_work_records_manifest_and_migrates(make_collection):
    import serve
    col = make_collection(files=("volcanoes.txt",), linked=False)
    col = _to_stem_layout(col, ("volcanoes.txt",))
    data = (CORPUS / "report.md").read_bytes()
    out = serve.import_work(col, "report.md", data)
    assert out["chunks"] > 0
    col = library.get(col.id)
    assert col.meta["schema"] == 2
    assert (col.chunks_dir / "volcanoes.txt.jsonl").exists()
    assert (col.chunks_dir / "report.md.jsonl").exists()
    m = ingest.load_manifest(col)
    assert m["report.md"]["src_sha256"] == ingest.file_hash(
        col.sources_dir / "report.md")
    ev = _stage(reingest(col), "chunk")
    assert ev and all(e["status"] == "up to date" for e in ev)
    # a same-stem upload re-chunks its sibling so ids stay unique
    serve.import_work(col, "report.txt", (CORPUS / "report.txt").read_bytes())
    ids = [c["id"] for c in rag.load_chunks(library.get(col.id))]
    assert len(ids) == len(set(ids))
    assert all(e["status"] == "up to date"
               for e in _stage(reingest(col), "chunk"))
