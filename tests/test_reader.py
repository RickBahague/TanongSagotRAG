"""Source reader: stitched passages reproduce the original text."""
import rag
from helpers import CORPUS


def _rt(make_collection, **kw):
    col = make_collection(files=("volcanoes.txt",), **kw)
    return rag.Runtime(col)


def test_full_passage_round_trip(make_collection):
    rt = _rt(make_collection, settings={"CHUNK_WORDS": 50,
                                        "CHUNK_OVERLAP_WORDS": 10})
    seq = rt.groups[("volcanoes.txt", rt.index.chunks[0]["essay"])]
    assert len(seq) == 3
    original = (CORPUS / "volcanoes.txt").read_text().split()
    for i, c in enumerate(seq):
        p = rt.passage(c["id"], before=len(seq), after=len(seq))
        stitched = " ".join([p["before"], p["cited"], p["after"]]).split()
        assert stitched == original, f"cited chunk {i}"
        assert p["has_more_before"] is False and p["has_more_after"] is False


def test_windowed_passage_flags(make_collection):
    rt = _rt(make_collection, settings={"CHUNK_WORDS": 50,
                                        "CHUNK_OVERLAP_WORDS": 10})
    seq = rt.groups[("volcanoes.txt", rt.index.chunks[0]["essay"])]
    p = rt.passage(seq[1]["id"], before=0, after=0)
    assert p["before"] == "" and p["after"] == ""
    assert p["cited"].split() == seq[1]["text"].split()
    assert p["has_more_before"] and p["has_more_after"]


def test_paragraph_breaks_restored(make_collection):
    col = make_collection(files=("lighthouses.html",))
    rt = rag.Runtime(col)
    c = rt.index.chunks[0]
    p = rt.passage(c["id"])
    assert p["cited"].count("\n\n") >= 2      # three <p> paragraphs
    assert " ".join(p["cited"].split()) == c["text"]
