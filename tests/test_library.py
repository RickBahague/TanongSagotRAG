"""Registry CRUD and collection isolation."""
import json

import pytest

import conversations
import library
import rag


def test_create_list_get_delete(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("hello " * 50)
    col = library.create("My Docs", link=str(folder), description="d",
                         author="Ann")
    assert col.id == "my-docs"
    assert [c.id for c in library.list_all()] == ["my-docs"]
    got = library.get("my-docs")
    assert (got.name, got.description, got.author, got.linked) == \
        ("My Docs", "d", "Ann", True)
    assert got.sources_dir == folder.resolve()
    library.delete("my-docs")
    assert library.list_all() == []
    with pytest.raises(KeyError):
        library.get("my-docs")
    assert (folder / "a.txt").exists(), "linked folder must never be touched"


def test_slug_dedupes():
    a = library.create("Notes")
    b = library.create("Notes")
    c = library.create("Notes!!")
    assert (a.id, b.id, c.id) == ("notes", "notes-2", "notes-3")


def test_get_sanitizes_id():
    library.create("Safe")
    # path characters are stripped, so traversal resolves inside the registry
    assert library.get("../safe").root == library.COLLECTIONS_DIR / "safe"
    with pytest.raises(KeyError):
        library.get("../../etc")


def test_create_rejects_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        library.create("x", link=str(tmp_path / "nope"))


def test_managed_collection_has_sources_dir():
    col = library.create("Managed")
    assert not col.linked
    assert col.sources_dir.is_dir()
    assert col.sources_dir.parent == col.root


def test_persona_requires_author():
    a = library.create("A", author="", persona={"enabled": True})
    b = library.create("B", author="Bea", persona={"enabled": True})
    assert a.persona is None
    assert b.persona == {"enabled": True}
    assert rag.available_modes(a) == ["scholar", "quotes"]
    assert rag.available_modes(b) == ["scholar", "quotes", "persona"]


def test_setting_override(make_collection):
    col = make_collection(files=("volcanoes.txt",), ingest=False,
                          settings={"TOP_K": 3})
    assert col.setting("TOP_K") == 3
    assert col.setting("BM25_K1") == 1.5


def test_two_collections_isolated(make_collection):
    bees = make_collection("Bees", files=("beekeeping.md",))
    ice = make_collection("Ice", files=("glaciers.docx",))
    assert bees.data_dir != ice.data_dir
    bee_src = {c["source"] for c in rag.load_chunks(bees)}
    ice_src = {c["source"] for c in rag.load_chunks(ice)}
    assert bee_src == {"beekeeping.md"} and ice_src == {"glaciers.docx"}

    bi, ii = rag.load_index(bees), rag.load_index(ice)
    assert rag.gate_ok(bi, "How do worker bees build the wax comb?")
    assert not rag.gate_ok(ii, "How do worker bees build the wax comb?")

    conversations.append_turn(bees, "c1", "q", "scholar", "a", [])
    assert [c["id"] for c in conversations.list_all(bees)] == ["c1"]
    assert conversations.list_all(ice) == []

    meta = json.loads((bees.root / "collection.json").read_text())
    assert "beekeeping.md" in meta["source_meta"]
    assert "beekeeping.md" not in ice.source_meta
