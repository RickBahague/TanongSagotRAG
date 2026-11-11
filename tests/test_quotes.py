"""Quote extraction + mechanical verbatim verification, and the quote bank."""
import sys

import rag
from helpers import FIXTURES

sys.path.insert(0, str(FIXTURES))
from build import QUOTE  # noqa: E402

CHUNKS = [{"id": "c1", "text": "Intro words here. " + QUOTE + " More text "
           "follows about propolis and the colony."},
          {"id": "c2", "text": "Glaciers carve U-shaped valleys and leave "
           "moraines behind when they retreat."}]


def _answer(*quotes):
    return "Framing sentence.\n\n" + "\n\n".join(
        f"> {q} [Hive Structure, p. 1]" for q in quotes)


def test_extract_quotes_strips_citation():
    qs = rag.extract_quotes(_answer(QUOTE))
    assert qs == [QUOTE]


def test_verbatim_quote_verified():
    [r] = rag.verify_quotes(_answer(QUOTE), CHUNKS)
    assert r["ok"] and r["chunk_id"] == "c1"


def test_typographic_variants_verified():
    q = QUOTE.replace("'", "’")
    [r] = rag.verify_quotes(_answer(q.upper()), CHUNKS)   # case + quotes
    assert r["ok"]


def test_altered_quote_rejected():
    altered = QUOTE.replace("feeding chosen larvae a diet of royal jelly",
                            "by importing a mated queen from a breeder")
    [r] = rag.verify_quotes(_answer(altered), CHUNKS)
    assert not r["ok"]


def test_fabricated_quote_rejected():
    [r] = rag.verify_quotes(_answer("Bees are known to migrate south every "
                                    "autumn in enormous flocks."), CHUNKS)
    assert not r["ok"]


def test_ellipsis_segments_each_verified():
    a, b = QUOTE.split(", feeding")
    ok = rag.verify_quotes(_answer(a + " … feeding" + b), CHUNKS)
    assert ok[0]["ok"]
    bad = rag.verify_quotes(_answer(a + " … and then flies to the moon "
                                    "with the drones"), CHUNKS)
    assert not bad[0]["ok"]


def test_quote_bank_entries_are_verbatim(make_collection):
    col = make_collection(files=("beekeeping.md", "volcanoes.txt"))
    rt = rag.Runtime(col)
    assert rt.quote_bank
    for e in rt.quote_bank:
        assert e["text"] in rt.by_id[e["chunk_id"]]["text"]
    q = rt.random_quote(topic="volcano eruption")
    assert q and q["text"] in rt.by_id[q["chunk_id"]]["text"]
