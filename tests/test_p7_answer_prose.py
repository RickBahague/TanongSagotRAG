"""Answer prose (plan.md "Answer prose"): compose_answer turns the small
model's one-block, excerpt-by-excerpt answers into paragraphs with bare
[n] citations, and the server/CLI deliver that form."""
import shutil

import pytest

import rag
from helpers import CLEAN_FILES, CORPUS

# real JMS answers from the desktop click-through (2026-11-20), shortened
WALK = ("The basic problems of the Philippine society, as discussed in "
        "Excerpt [1], include imperialism, feudalism, and bureaucrat "
        "capitalism. Excerpt [2] elaborates on how these problems have "
        "persisted and can be changed by the revolutionary forces. Excerpt "
        "[4] mentions that the semicolonial and semifeudal character of "
        "Philippine society has persisted. Excerpt [5] outlines the program "
        "for a people's democratic revolution.")

LOOP = "\n\n".join(
    ["The peasantry is the main force of the people's democratic "
     "revolution. [2]",
     "The peasantry is the most reliable ally of the proletariat. [1]",
     "The peasantry is the main force of the people's democratic "
     "revolution. [2]",
     "The poor peasants and farm workers comprise about 75 percent of the "
     "rural population. [6]"])


def _paras(text):
    return text.split("\n\n")


# ------------------------------------------------------------- narration --

@pytest.mark.parametrize("raw,want", [
    ("The problems, as discussed in Excerpt [1], include imperialism.",
     "The problems include imperialism [1]."),
    ("Excerpt [2] elaborates on how these problems have persisted.",
     "These problems have persisted [2]."),
    ("Excerpt [4] mentions that the character has persisted.",
     "The character has persisted [4]."),
    ("According to excerpt 3, land was seized.", "Land was seized [3]."),
    ("In Excerpt [2], the strike is described.", "The strike is described [2]."),
    ("Excerpt [3] discusses the propagation of the theory.",
     "The source discusses the propagation of the theory [3]."),
    ("Excerpts 2 and 4 describe the strike.",
     "The sources describe the strike [2][4]."),
    ("Land was seized (excerpt 3).", "Land was seized [3]."),
    ("The need is also emphasized in the excerpt, stating that it is needed.",
     "The need is also emphasized, stating that it is needed."),
    ("The basic problems, as discussed in the excerpts, include imperialism.",
     "The basic problems include imperialism."),
    ("The excerpted passage is long [1].", "The excerpted passage is long [1]."),
])
def test_narration_becomes_a_marker(raw, want):
    assert rag.compose_answer(raw) == want


def test_walk_through_excerpts_rewritten():
    out = rag.compose_answer(WALK)
    assert "xcerpt" not in out
    assert out.startswith("The basic problems of the Philippine society "
                          "include imperialism, feudalism, and bureaucrat "
                          "capitalism [1].")
    assert "These problems have persisted and can be changed" in out
    assert rag.citation_markers(out) == [1, 2, 4, 5]


# ---------------------------------------------------------------- markers --

@pytest.mark.parametrize("raw,want", [
    ("Bees build comb. [1]", "Bees build comb [1]."),
    ("Bees build comb. [2]. [1]", "Bees build comb [2][1]."),
    ("[2]. Bees build comb.", "Bees build comb [2]."),
    ("Bees build comb [1][1].", "Bees build comb [1]."),
    ("Bees build comb [1].", "Bees build comb [1]."),
    ("Bees build comb.", "Bees build comb."),
    ("The U.S. Navy left. [3]", "The U.S. Navy left [3]."),
])
def test_marker_before_full_stop(raw, want):
    assert rag.compose_answer(raw) == want


def test_never_invents_citations_and_drops_repeats():
    out = rag.compose_answer(LOOP)
    assert set(rag.citation_markers(out)) <= {1, 2, 6}
    assert out.count("main force of the people's democratic revolution") == 1


# ------------------------------------------------------------- paragraphs --

def test_one_block_split_at_topic_change():
    raw = ("Worker bees secrete wax from glands on their abdomen [1]. The "
           "wax is chewed and shaped into hexagonal comb cells [1]. The "
           "cells hold brood, pollen and honey [1]. Lighthouses use Fresnel "
           "lenses to focus the beam [2]. The lens concentrates light into "
           "a narrow beam that sailors see far out at sea [2].")
    paras = _paras(rag.compose_answer(raw))
    assert len(paras) == 2
    assert paras[0].startswith("Worker bees") and paras[1].startswith("Lighthouses")


def test_continuation_sentence_stays_in_paragraph():
    raw = ("Worker bees secrete wax [1]. Comb cells are hexagonal [1]. Brood "
           "is raised in the cells [1]. It is fed by nurse bees [2].")
    assert len(_paras(rag.compose_answer(raw))) == 1


def test_long_block_capped():
    raw = " ".join(f"Bees store honey in comb cell number {i} [1]."
                   for i in range(12))
    paras = _paras(rag.compose_answer(raw))
    assert all(len(rag._split_sentences(p)) <= rag.PARA_MAX_SENTENCES
               for p in paras)


def test_one_sentence_lines_are_merged():
    out = rag.compose_answer(LOOP)
    assert len(_paras(out)) < 3


def test_model_paragraphs_kept():
    raw = ("Worker bees secrete wax [1]. They shape it into cells [1].\n\n"
           "Honey is stored in the upper comb [2]. Brood sits lower [2].")
    assert rag.compose_answer(raw) == raw


def test_markdown_left_alone():
    raw = ("Three causes [1]:\n\n- imperialism [1]\n- feudalism [2]\n\n"
           "```\ncode. [1]\n```")
    assert rag.compose_answer(raw) == raw
    lst = "Three causes [1].\n\n- imperialism. [1]\n- feudalism [2]"
    assert rag.compose_answer(lst).endswith("- imperialism. [1]\n- feudalism [2]")


@pytest.mark.parametrize("raw", [WALK, LOOP, "", "Short.",
                                 "Bees build comb. [2]. [1] It works."])
def test_idempotent(raw):
    once = rag.compose_answer(raw)
    assert rag.compose_answer(once) == once


# --------------------------------------------------------------- delivery --

def test_server_sends_and_saves_composed_answer(server, fake_llm, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for f in CLEAN_FILES:
        shutil.copy2(CORPUS / f, d / f)
    st, col = server.json("POST", "/collections", {"name": "P", "link": str(d)})
    cid = col["id"]
    server.ndjson("POST", f"/c/{cid}/ingest?summaries=0")
    fake_llm.reply = ("Excerpt [1] mentions that worker bees secrete wax. "
                      "They shape it into comb. [1]")
    st, evs = server.ndjson("POST", f"/c/{cid}/ask",
                            {"question": "How do worker bees build the wax "
                                         "comb?", "mode": "scholar"})
    final = [e for e in evs if e["type"] == "final"]
    want = "Worker bees secrete wax [1]. They shape it into comb [1]."
    assert final and final[0]["text"] == want
    assert [e["type"] for e in evs][-1] == "done"
    cites = [e for e in evs if e["type"] == "citations"][0]
    assert cites["used"] == [1]
    st, turns = server.json("GET", f"/c/{cid}/conversation/"
                            + evs[0]["conversation_id"])
    assert turns[0]["answer"] == want


def test_blocking_ask_composes(make_collection, fake_llm):
    col = make_collection()
    fake_llm.reply = "Excerpt [1] says that bees build comb."
    res = rag.ask(col, "How do worker bees build the wax comb?")
    assert res["answer"] == "Bees build comb [1]."


# ---------------------------------------------------- lifted sentences --

EXCERPTS = [
    "Lighthouses use Fresnel lenses to focus the lamp into a beam.",
    "Worker bees secrete wax from glands on the abdomen and chew it into "
    "hexagonal comb cells.",
    "Glaciers form where snow accumulates faster than it melts.",
]


def test_uncited_lifted_sentence_gets_its_excerpt():
    raw = "Worker bees secrete wax from abdomen glands and chew it into comb cells."
    assert rag.compose_answer(raw, EXCERPTS).endswith("comb cells [2].")


def test_wrong_number_on_lifted_sentence_corrected():
    raw = ("Worker bees secrete wax from abdomen glands and chew it into "
           "comb cells [1].")
    assert rag.compose_answer(raw, EXCERPTS).endswith("comb cells [2].")


def test_paraphrase_and_plausible_cites_kept():
    para = "Bees make their homes from a substance their bodies produce [1]."
    assert rag.compose_answer(para, EXCERPTS) == para       # not lifted
    both = ("Worker bees secrete wax from glands and chew it into comb "
            "cells [3][2].")
    assert rag.compose_answer(both, EXCERPTS) == both       # [2] is right
    short = "Bees chew wax."                                # too few words
    assert rag.compose_answer(short, EXCERPTS) == short
    titled = ("Worker bees secrete wax from abdomen glands and chew it into "
              "comb cells [Hive Structure, p. 1].")
    assert rag.compose_answer(titled, EXCERPTS) == titled


def test_lifted_idempotent():
    raw = ("Worker bees secrete wax from abdomen glands and chew it into "
           "comb cells [1]. Glaciers form where snow accumulates faster "
           "than it melts.")
    once = rag.compose_answer(raw, EXCERPTS)
    assert rag.citation_markers(once) == [2, 3]
    assert rag.compose_answer(once, EXCERPTS) == once
