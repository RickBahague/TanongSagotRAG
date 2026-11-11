"""HTTP smoke test through serve.make_server(0) with the fake LLM."""
import shutil

from helpers import CLEAN_FILES, CORPUS


def _linked_folder(tmp_path):
    d = tmp_path / "web-docs"
    d.mkdir()
    for f in CLEAN_FILES:
        shutil.copy2(CORPUS / f, d / f)
    return d


def test_full_flow(server, fake_llm, tmp_path):
    folder = _linked_folder(tmp_path)
    st, col = server.json("POST", "/collections",
                          {"name": "Web Test", "link": str(folder)})
    assert st == 200 and col["sources"] == len(CLEAN_FILES)
    cid = col["id"]

    st, events = server.ndjson("POST", f"/c/{cid}/ingest?summaries=0")
    assert st == 200
    assert events[-1]["stage"] == "complete", events[-1]
    assert not [e for e in events if e["stage"] == "error"]

    st, stats = server.json("GET", f"/c/{cid}/stats")
    assert st == 200 and len(stats["works"]) == len(CLEAN_FILES)
    assert stats["modes"] == ["scholar", "quotes"]
    assert stats["retrieval"] == "BM25Index"

    st, cols = server.json("GET", "/collections")
    assert [c["id"] for c in cols] == [cid] and cols[0]["indexed"]

    fake_llm.reply = "Worker bees secrete wax [Hive Structure, p. 1]."
    st, evs = server.ndjson("POST", f"/c/{cid}/ask",
                            {"question": "How do worker bees build the wax "
                                         "comb?", "mode": "scholar"})
    assert st == 200
    types = [e["type"] for e in evs]
    assert types[0] == "meta" and types[-1] == "done" and "delta" in types
    meta = evs[0]
    assert meta["sources"] and meta["sources"][0]["essay"] == "Hive Structure"
    answer = "".join(e["text"] for e in evs if e["type"] == "delta")
    assert answer == fake_llm.reply
    assert "Worker bees secrete wax" in fake_llm.prompt_text()

    conv = meta["conversation_id"]
    st, convs = server.json("GET", f"/c/{cid}/conversations")
    assert [c["id"] for c in convs] == [conv]
    st, turns = server.json("GET", f"/c/{cid}/conversation/{conv}")
    assert turns[0]["answer"] == answer
    st, md = server.request("GET", f"/c/{cid}/conversation/{conv}/export")
    assert st == 200 and b"## How do worker bees" in md

    st, p = server.json("GET", f"/c/{cid}/passage?id="
                        + _q(meta["sources"][0]["id"]))
    assert st == 200 and p["essay"] == "Hive Structure" and p["cited"]

    st, _ = server.json("POST", f"/c/{cid}/source-meta",
                        {"filename": "volcanoes.txt", "title": "Fire Mountains"})
    assert st == 200
    st, stats = server.json("GET", f"/c/{cid}/stats")
    assert "Fire Mountains" in {w["title"] for w in stats["works"]}

    st, _ = server.json("DELETE", f"/collections/{cid}")
    assert st == 200
    st, cols = server.json("GET", "/collections")
    assert cols == []
    assert (folder / "volcanoes.txt").exists()


def test_quotes_mode_emits_verify(server, fake_llm, tmp_path):
    import sys
    from helpers import FIXTURES
    sys.path.insert(0, str(FIXTURES))
    from build import QUOTE
    folder = _linked_folder(tmp_path)
    _, col = server.json("POST", "/collections",
                         {"name": "Q", "link": str(folder)})
    server.ndjson("POST", f"/c/{col['id']}/ingest?summaries=0")
    fake_llm.reply = (f"Framing.\n\n> {QUOTE} [Hive Structure, p. 1]\n\n"
                      "> Bees hibernate in caves all winter long like bears. "
                      "[Hive Structure, p. 1]")
    _, evs = server.ndjson("POST", f"/c/{col['id']}/ask",
                           {"question": "What happens when a colony loses "
                                        "its queen?", "mode": "quotes"})
    [verify] = [e for e in evs if e["type"] == "verify"]
    assert [q["ok"] for q in verify["quotes"]] == [True, False]


def test_gated_ask_needs_no_llm(server, tmp_path):
    # LLM stays pointed at a dead port: a gated question must still answer
    folder = _linked_folder(tmp_path)
    _, col = server.json("POST", "/collections",
                         {"name": "G", "link": str(folder)})
    server.ndjson("POST", f"/c/{col['id']}/ingest?summaries=0")
    _, evs = server.ndjson("POST", f"/c/{col['id']}/ask",
                           {"question": "Recipe for chocolate cake"})
    assert [e["type"] for e in evs] == ["meta", "delta", "done"]
    assert "do not appear to cover" in evs[1]["text"]


def test_unreachable_backend_is_stream_error(server, tmp_path):
    folder = _linked_folder(tmp_path)
    _, col = server.json("POST", "/collections",
                         {"name": "U", "link": str(folder)})
    server.ndjson("POST", f"/c/{col['id']}/ingest?summaries=0")
    _, evs = server.ndjson("POST", f"/c/{col['id']}/ask",
                           {"question": "What did Fresnel invent for "
                                        "lighthouses?"})
    assert evs[-1]["type"] == "error"
    assert "Cannot reach" in evs[-1]["error"]


def test_import_into_managed_collection(server):
    _, col = server.json("POST", "/collections", {"name": "Managed"})
    st, res = server.json("POST", f"/c/{col['id']}/import",
                          (CORPUS / "beekeeping.md").read_bytes(),
                          headers={"X-Filename": "beekeeping.md"})
    assert st == 200, res
    assert res["chunks"] == 3 and res["segmentation"] == "headings"


def test_unknown_routes(server):
    assert server.json("GET", "/c/nope/stats")[0] == 404
    assert server.json("GET", "/nope")[0] == 404
    assert server.json("POST", "/c/nope/ask", {"question": "x"})[0] == 404


def _q(s):
    from urllib.parse import quote
    return quote(s)
