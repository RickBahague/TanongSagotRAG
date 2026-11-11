"""M1-A: prune events, chunk-param staleness, schema keys, atomic writes."""
import json

import ingest
import library
import rag
from helpers import CORPUS, reingest


def test_prune_emits_event_and_keeps_other_sources(make_collection):
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    # user-owned files in data/ that must survive a prune
    (col.data_dir / "eval.jsonl").write_text('{"q": "x", "expect": "y"}\n')
    (col.conversations_dir / "volcanoes-chat.json").write_text("[]")
    keep = {p: p.read_bytes() for p in col.data_dir.rglob("beekeeping*")}
    assert keep

    (col.sources_dir / "volcanoes.txt").unlink()
    events = reingest(col)

    prune = [e for e in events if e["stage"] == "prune"]
    assert len(prune) == 1 and prune[0]["source"] == "volcanoes.txt"
    assert prune[0]["removed"] >= 4            # chunks, paras, dates, quotes
    assert events.index(prune[0]) < min(
        i for i, e in enumerate(events) if e["stage"] == "chunk")
    for p, data in keep.items():
        assert p.read_bytes() == data
    assert (col.data_dir / "eval.jsonl").exists()
    assert (col.conversations_dir / "volcanoes-chat.json").exists()
    assert (col.sources_dir / "beekeeping.md").exists()
    assert set(ingest.load_manifest(col)) == {"beekeeping.md"}
    # nothing left to prune on the next run
    assert not [e for e in reingest(col) if e["stage"] == "prune"]


def test_prune_removes_summary_and_vector_artifacts(make_collection):
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    for p in (col.summaries_dir / "volcanoes.txt.jsonl",
              col.summaries_dir / "volcanoes.txt.jsonl.src",
              col.vectors_dir / "volcanoes.txt.npz",
              col.vectors_dir / "volcanoes.txt.summaries.npz",
              col.vectors_dir / "beekeeping.md.npz"):
        p.write_text("x")
    (col.sources_dir / "volcanoes.txt").unlink()
    reingest(col)
    assert not list(col.data_dir.rglob("volcanoes*"))
    assert (col.vectors_dir / "beekeeping.md.npz").exists()


def test_missing_linked_folder_does_not_prune(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    folder = col.sources_dir
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()                             # e.g. unmounted drive
    events = reingest(col)
    assert events[-1]["stage"] == "error"
    assert (col.chunks_dir / "volcanoes.txt.jsonl").exists()


def test_chunk_settings_change_triggers_rechunk(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    entry = ingest.load_manifest(col)["volcanoes.txt"]
    assert entry["chunk_params"] == {"chunk_words": 450, "overlap_words": 80}
    assert entry["src_sha256"] == ingest.file_hash(
        CORPUS / "volcanoes.txt")
    n_before = entry["n_chunks"]

    col.settings_file.write_text(json.dumps({"CHUNK_WORDS": 50,
                                             "CHUNK_OVERLAP_WORDS": 10}))
    ev = [e for e in reingest(col) if e["stage"] == "chunk"]
    assert ev[0]["status"] != "up to date"
    entry = ingest.load_manifest(col)["volcanoes.txt"]
    assert entry["chunk_params"] == {"chunk_words": 50, "overlap_words": 10}
    assert entry["n_chunks"] > n_before
    # unchanged settings + content → up to date again
    ev = [e for e in reingest(col) if e["stage"] == "chunk"]
    assert ev[0]["status"] == "up to date"


def test_every_word_kept_with_small_overlap(make_collection):
    col = make_collection(files=("volcanoes.txt",),
                          settings={"CHUNK_WORDS": 40,
                                    "CHUNK_OVERLAP_WORDS": 5})
    words = (CORPUS / "volcanoes.txt").read_text().split()
    chunks = rag.load_chunks(col)
    covered = set()
    for c in chunks:
        start = int(c["id"].rsplit("-", 1)[1])
        covered.update(range(start, start + len(c["text"].split())))
    assert covered == set(range(len(words)))


def test_schema_keys_present(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    meta = json.loads((col.root / "collection.json").read_text())
    assert meta["schema"] == library.COLLECTION_SCHEMA == 2
    raw = json.loads(col.manifest_file.read_text())
    assert raw["_schema"] == ingest.MANIFEST_SCHEMA == 2
    # readers see only sources
    assert set(ingest.load_manifest(col)) == {"volcanoes.txt"}
    stats_total = sum(e.get("n_chunks", 0)
                      for e in ingest.load_manifest(col).values())
    assert stats_total == len(rag.load_chunks(col))


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    library.write_json_atomic(target, {"a": 1})
    library.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_derived_paths_single_mapping(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    paths = col.derived_paths("volcanoes.txt")
    assert all(col.data_dir in p.parents for p in paths)
    on_disk = col.derived_files_on_disk()
    assert set(on_disk) <= set(paths)
    assert all(keys == {"volcanoes.txt"} for keys in on_disk.values())
