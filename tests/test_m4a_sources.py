"""M4-A (P3-2): PATCH /c/<id>, source removal, single-source re-ingest,
recursive linked-folder scan with ignore patterns, manifest error/no_text
flags in /stats, and disk usage in /collections."""
import json
import shutil
from urllib.parse import quote

import ingest
import library
from helpers import CLEAN_FILES, CORPUS


def _blank_pdf(path, pages=2):
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        w.write(f)


def _ingest(server, cid, query="summaries=0"):
    st, evs = server.ndjson("POST", f"/c/{cid}/ingest?{query}")
    assert st == 200, evs
    return evs


# --------------------------------------------------------------- PATCH --

def test_patch_collection(server, make_collection):
    col = make_collection(ingest=False)
    st, res = server.json("PATCH", f"/c/{col.id}",
                          {"name": "  Renamed  ", "description": "Docs",
                           "author": "Ada Lovelace",
                           "persona": {"enabled": True}})
    assert st == 200, res
    assert res["name"] == "Renamed" and res["author"] == "Ada Lovelace"
    assert res["persona"] == {"enabled": True} and res["id"] == col.id
    got = library.get(col.id)
    assert (got.name, got.description, got.author) == \
        ("Renamed", "Docs", "Ada Lovelace") and got.persona
    reg = json.loads(library.REGISTRY.read_text())
    assert reg[0]["name"] == "Renamed"
    st, cols = server.json("GET", "/collections")
    assert cols[0]["name"] == "Renamed"

    st, res = server.json("PATCH", f"/c/{col.id}", {"persona": None})
    assert st == 200 and library.get(col.id).persona is None


def test_patch_persona_mode_live(server, make_collection):
    col = make_collection()
    st, s = server.json("GET", f"/c/{col.id}/stats")
    assert "persona" not in s["modes"]
    server.json("PATCH", f"/c/{col.id}", {"author": "Jane Doe",
                                         "persona": {"enabled": True}})
    st, s = server.json("GET", f"/c/{col.id}/stats")
    assert "persona" in s["modes"] and s["author"] == "Jane Doe"


def test_patch_validation(server, make_collection):
    col = make_collection(ingest=False)
    for body, key in (({"name": ""}, "name"), ({"name": 3}, "name"),
                      ({"description": "x" * 3000}, "description"),
                      ({"link": "/tmp"}, "link"), ({"id": "x"}, "id"),
                      ({"persona": "yes"}, "persona"),
                      ({"persona": {"enabled": "1"}}, "persona.enabled"),
                      ({"persona": {"enabled": True, "backend": "x"}},
                       "persona.backend"),
                      ({"persona": {"zzz": 1}}, "persona.zzz")):
        st, res = server.json("PATCH", f"/c/{col.id}", dict(body, author="A"))
        assert st == 400, (body, res)
        assert key in res["errors"], (body, res)
    assert library.get(col.id).author == ""          # nothing written
    st, _ = server.request("PATCH", f"/c/{col.id}", b'{"name": "x"}',
                           headers={"Content-Type": "text/plain"})
    assert st == 415
    st, _ = server.json("PATCH", "/c/nope", {"name": "x"})
    assert st == 404
    st, _ = server.request("PATCH", f"/c/{col.id}", {"name": "x"},
                           headers={"Origin": "http://evil.example"})
    assert st == 403
    assert library.get(col.id).name == "Fixtures"


# -------------------------------------------------------- source removal --

def test_delete_linked_source_refused(server, make_collection):
    col = make_collection()
    target = col.sources_dir / "volcanoes.txt"
    before = target.read_bytes()
    st, res = server.json("DELETE", f"/c/{col.id}/source/volcanoes.txt")
    assert st == 409 and "never deletes" in res["error"]
    assert target.read_bytes() == before
    assert col.chunk_file("volcanoes.txt").exists()


def test_delete_managed_source(server, make_collection):
    col = make_collection(linked=False)
    name = "beekeeping.md"
    derived = [p for p in col.derived_paths(name) if p.exists()]
    assert col.chunk_file(name) in derived and len(derived) >= 3
    others = {p for p in col.derived_files_on_disk()} - set(derived)
    st, res = server.json("DELETE", f"/c/{col.id}/source/{quote(name)}")
    assert st == 200, res
    assert res["source"] == name and res["deleted"] == len(derived)
    assert res["indexed"] is True
    assert not (col.sources_dir / name).exists()
    assert not any(p.exists() for p in derived)
    assert all(p.exists() for p in others)            # siblings untouched
    assert name not in ingest.load_manifest(col)
    assert name not in library.get(col.id).source_meta
    st, s = server.json("GET", f"/c/{col.id}/stats")
    assert name not in {w["source"] for w in s["works"]}
    assert len(s["works"]) == len(CLEAN_FILES) - 1

    st, _ = server.json("DELETE", f"/c/{col.id}/source/{quote(name)}")
    assert st == 404
    st, _ = server.json("DELETE", f"/c/{col.id}/source/..%2F..%2Fcollection.json")
    assert st == 404 and (col.root / "collection.json").exists()


def test_delete_last_managed_source(server, make_collection):
    col = make_collection(linked=False, files=("volcanoes.txt",))
    st, res = server.json("DELETE", f"/c/{col.id}/source/volcanoes.txt")
    assert st == 200 and res["indexed"] is False
    st, _ = server.json("GET", f"/c/{col.id}/stats")
    assert st == 409


# ------------------------------------------------- single-source ingest --

def test_single_source_reingest(server, make_collection):
    col = make_collection()
    before = ingest.load_manifest(col)
    evs = _ingest(server, col.id, "summaries=0&source=beekeeping.md")
    assert evs[-1]["stage"] == "complete", evs[-1]
    chunk = [e for e in evs if e["stage"] == "chunk"]
    assert [e["source"] for e in chunk] == ["beekeeping.md"]
    assert chunk[0]["done"] == 1 and chunk[0]["total"] == 1
    assert "chunks" in chunk[0]["status"]               # forced, not skipped
    embed = {e["source"] for e in evs if e["stage"] == "embed"}
    assert embed <= {"beekeeping.md"}
    after = ingest.load_manifest(col)
    for name in CLEAN_FILES:
        if name != "beekeeping.md":
            assert after[name] == before[name]
    st, res = server.json("POST", f"/c/{col.id}/ingest?source=nope.md")
    assert st == 404 and "nope.md" in res["error"]


# ----------------------------------------------------- recursive scan --

def test_recursive_linked_scan(server, tmp_path):
    root = tmp_path / "tree"
    for sub in ("a", "b", "a/deep", ".hidden", "node_modules", "__pycache__"):
        (root / sub).mkdir(parents=True)
    (root / "report.md").write_text("# Top\n\nTop level report about "
                                    "orchards and apple harvests.\n")
    (root / "a" / "report.md").write_text(
        "Report A covers submarine cables laid across the ocean floor.")
    (root / "b" / "report.md").write_text(
        "Report B covers falconry and the training of hunting hawks.")
    (root / "a" / "deep" / "notes.txt").write_text(
        "Deep notes about the history of chess openings.")
    for ignored in (".hidden/x.md", "node_modules/y.md", "__pycache__/z.txt",
                    "a/.secret.md", "~$draft.md", ".DS_Store", "a/tmp.tmp",
                    "a/unsupported.xyz"):
        (root / ignored).write_text("ignored zebra content")
    st, c = server.json("POST", "/collections", {"name": "Tree",
                                                 "link": str(root)})
    assert st == 200 and c["sources"] == 4
    col = library.get(c["id"])
    names = [col.source_name(p) for p in col.source_files()]
    assert names == ["a/deep/notes.txt", "a/report.md", "b/report.md",
                     "report.md"]
    keys = {col.artifact_key(n) for n in names}
    assert len(keys) == 4 and "report.md" in keys     # flat key unchanged
    assert all("/" not in k for k in keys)

    evs = _ingest(server, col.id)
    assert evs[-1]["stage"] == "complete", evs[-1]
    files = sorted(p.name for p in col.chunks_dir.glob("*.jsonl"))
    assert len(files) == 4
    srcs = set()
    for f in col.chunks_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            srcs.add(json.loads(line)["source"])
    assert srcs == set(names)
    assert set(ingest.load_manifest(col)) == set(names)

    st, s = server.json("GET", f"/c/{col.id}/stats")
    assert {w["source"] for w in s["works"]} == set(names)
    ids = set()
    for f in col.chunks_dir.glob("*.jsonl"):
        ids |= {json.loads(l)["id"] for l in f.read_text().splitlines()}
    assert len(ids) == sum(w["chunks"] for w in s["works"])   # ids unique

    # second run: everything up to date, nothing pruned
    evs = _ingest(server, col.id)
    assert not [e for e in evs if e["stage"] == "prune"]
    assert {e["status"] for e in evs if e["stage"] == "chunk"} == \
        {"up to date"}
    # an ignored path never becomes a source
    assert not [n for n in names if "zebra" in n]


# ------------------------------------------------ error / no_text flags --

def test_no_text_pdf_flag(server, make_collection):
    col = make_collection(files=("volcanoes.txt",), ingest=False)
    _blank_pdf(col.sources_dir / "scanned.pdf")
    evs = _ingest(server, col.id)
    assert evs[-1]["stage"] == "complete", evs[-1]
    ev = [e for e in evs if e["stage"] == "chunk"
          and e["source"] == "scanned.pdf"][0]
    assert ev["no_text"] is True and "OCR" in ev["status"]
    m = ingest.load_manifest(col)
    assert m["scanned.pdf"]["no_text"] is True
    assert "no_text" not in m["volcanoes.txt"]
    st, s = server.json("GET", f"/c/{col.id}/stats")
    works = {w["source"]: w for w in s["works"]}
    assert works["scanned.pdf"]["no_text"] is True
    assert works["scanned.pdf"]["chunks"] == 0
    assert works["volcanoes.txt"]["no_text"] is False
    assert works["volcanoes.txt"]["error"] is None


def test_error_field_in_stats(server, make_collection):
    col = make_collection(files=("volcanoes.txt",), ingest=False)
    (col.sources_dir / "broken.docx").write_bytes(b"not a zip at all")
    evs = _ingest(server, col.id)
    assert evs[-1]["stage"] == "complete", evs[-1]
    st, s = server.json("GET", f"/c/{col.id}/stats")
    works = {w["source"]: w for w in s["works"]}
    assert works["broken.docx"]["error"]
    assert works["broken.docx"]["chunks"] == 0
    assert ingest.load_manifest(col)["broken.docx"]["error"]


# ------------------------------------------------------------ disk size --

def test_collections_size(server, make_collection):
    linked = make_collection()
    managed = make_collection(name="Managed", linked=False)
    st, cols = server.json("GET", "/collections")
    by = {c["id"]: c for c in cols}
    src = sum((CORPUS / f).stat().st_size for f in CLEAN_FILES)
    for col in (linked, managed):
        c = by[col.id]
        assert c["source_bytes"] == src
        assert c["size_bytes"] > 0 and c["ingesting"] is False
    # managed sources live inside the collection dir, linked ones do not
    assert by[managed.id]["size_bytes"] >= src
    assert by[managed.id]["size_bytes"] - by[linked.id]["size_bytes"] > \
        src // 2
