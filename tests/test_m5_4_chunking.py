"""M5-4 (P2-4): paragraph/sentence-aware chunk windows with explicit
start/end word offsets; passage() stitches with the offsets and still
reproduces the source exactly. Chunk files written before P2-4 (no
offsets, fixed windows) keep working."""
import json
import random

import pytest

import ingest
import library
import rag

SENT = ("The {a} survey crew mapped the {b} ridge in the early morning "
        "light before the wind rose.")
ADJ = ["northern", "quiet", "eastern", "steep", "old", "wet", "dry", "far"]


def _doc(n_paras=6, seed=3):
    rng = random.Random(seed)
    paras = []
    for _ in range(n_paras):
        k = rng.randint(2, 6)
        paras.append(" ".join(SENT.format(a=rng.choice(ADJ), b=rng.choice(ADJ))
                              for _ in range(k)))
    return "\n\n".join(paras) + "\n"


# ------------------------------------------------------------ chunk_spans --

@pytest.mark.parametrize("size,ov", [(50, 10), (40, 0), (30, 29), (450, 80),
                                     (20, 5)])
def test_spans_cover_everything_and_respect_size(size, ov):
    text = _doc(12)
    words = text.split()
    paras = ingest.words_with_paras([(1, text)])[2]
    spans = ingest.chunk_spans(words, size, ov, paras)
    assert spans[0][0] == 0 and spans[-1][1] == len(words)
    covered = set()
    for (s, e), nxt in zip(spans, spans[1:] + [None]):
        assert 0 < e - s <= size
        covered.update(range(s, e))
        if nxt:
            assert s < nxt[0] <= e                 # progress, no gap
            assert e - nxt[0] <= ov                # overlap never grows
    assert covered == set(range(len(words)))


def test_spans_prefer_paragraph_then_sentence():
    words = ("a " * 44 + "end. " + "b " * 3 + "c " * 60).split()
    # paragraph start at 47, sentence end at 45: paragraph wins
    assert ingest.chunk_spans(words, 50, 10, paras={47})[0] == (0, 47)
    assert ingest.chunk_spans(words, 50, 10, paras=set())[0] == (0, 45)
    # no boundary inside the tolerance: cut at size, as before P2-4
    assert ingest.chunk_spans(("x " * 120).split(), 50, 10)[0] == (0, 50)


def test_next_window_starts_at_sentence_inside_overlap():
    words = ("w " * 38 + "stop. " + "v " * 80).split()
    spans = ingest.chunk_spans(words, 50, 15, paras=set())
    # first window cut at 50 (no boundary in 45..50); the next window
    # starts at the sentence start 39 inside the 15-word overlap (35..49)
    assert spans[0] == (0, 50) and spans[1][0] == 39


def test_abbreviations_are_not_sentence_ends():
    words = "Mount St. Helens erupted. Dr. Smith wrote J. Doe. It ended!".split()
    starts = ingest.sentence_starts(words)
    assert starts == {4, 9, 11}


# --------------------------------------------------------- stored offsets --

@pytest.fixture
def small_col(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "ridge.txt").write_text(_doc(10))
    col = library.create("Chunk", link=str(d))
    col.settings_file.write_text(json.dumps({"CHUNK_WORDS": 50,
                                             "CHUNK_OVERLAP_WORDS": 10}))
    col = library.get(col.id)
    ingest.run_pipeline(col, summaries=False, workers=1,
                        progress=lambda e: None)
    return library.get(col.id)


def test_chunks_store_offsets(small_col):
    words = (small_col.source_path("ridge.txt").read_text()).split()
    chunks = rag.load_chunks(small_col)
    assert len(chunks) > 3
    for c in chunks:
        assert c["text"].split() == words[c["start"]:c["end"]]
        assert c["id"].endswith(f"-{c['start']}")
    # more windows end at a sentence than the old fixed windows did
    ends = sum(c["text"].endswith(".") for c in chunks)
    old = [ck for _, ck in ingest.chunk_words(words, 50, 10) if ck]
    old_ends = sum(ck[-1].endswith(".") for ck in old)
    assert ends / len(chunks) > old_ends / len(old)


def test_passage_round_trip_exact(small_col):
    original = small_col.source_path("ridge.txt").read_text().split()
    rt = rag.Runtime(small_col)
    seq = rt.groups[("ridge.txt", "ridge")]
    for c in seq:
        p = rt.passage(c["id"], before=len(seq), after=len(seq))
        stitched = " ".join([p["before"], p["cited"], p["after"]]).split()
        assert stitched == original, c["id"]
        assert "[…]" not in stitched
    # windowed: neighbours only, still no repeated or missing word
    for i in range(1, len(seq) - 1):
        p = rt.passage(seq[i]["id"], before=1, after=1)
        st = " ".join([p["before"], p["cited"], p["after"]]).split()
        assert st == original[seq[i - 1]["start"]:seq[i + 1]["end"]]


def test_paragraph_breaks_survive(small_col):
    rt = rag.Runtime(small_col)
    seq = rt.groups[("ridge.txt", "ridge")]
    p = rt.passage(seq[0]["id"], before=0, after=len(seq))
    assert (p["cited"] + p["after"]).count("\n\n") >= 3


def test_gap_marker_when_a_chunk_is_missing(small_col):
    rt = rag.Runtime(small_col)
    seq = rt.groups[("ridge.txt", "ridge")]
    del seq[2]                                 # simulate a missing window
    p = rt.passage(seq[1]["id"], before=1, after=2)
    assert "[…]" in p["after"]


def test_legacy_fixed_window_chunks_still_stitch(small_col):
    """A chunk file from before P2-4: fixed 50/10 windows, no offsets."""
    words = small_col.source_path("ridge.txt").read_text().split()
    f = small_col.chunk_file("ridge.txt")
    first = json.loads(f.read_text().splitlines()[0])
    prefix = first["id"].rsplit("-", 2)[0]
    rows = [{"id": f"{prefix}-ridge-{s}", "text": " ".join(ck),
             "essay": "ridge", "page_start": 1, "page_end": 1,
             "source": "ridge.txt"}
            for s, ck in ingest.chunk_words(words, 50, 10) if ck]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rt = rag.Runtime(library.get(small_col.id))
    seq = rt.groups[("ridge.txt", "ridge")]
    assert all("start" not in c for c in seq)
    for c in seq:
        p = rt.passage(c["id"], before=len(seq), after=len(seq))
        st = " ".join([p["before"], p["cited"], p["after"]]).split()
        assert st == words


def test_old_chunk_files_rechunk_once(small_col):
    m = ingest.load_manifest(small_col)
    e = m["ridge.txt"]
    assert e["segmenter"] == ingest.SEGMENTER_VERSION == 3
    e["segmenter"] = 2                          # chunked by M5-3's code
    assert not ingest.chunk_is_current(
        small_col, small_col.source_path("ridge.txt"), e)
