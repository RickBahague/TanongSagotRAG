"""M5-5 (P2-6): Unicode tokenizer (NFKC + casefold + Unicode word chars)
and per-collection stopwords (<collection>/stopwords.txt).

The non-ASCII documents and their questions live here, not in CLEAN_FILES,
so the fixture yardstick (tests/eval_fixture.py) stays comparable."""
import json
import re

import pytest

import library
import ingest
import rag
from helpers import CLEAN_FILES, CORPUS, load_eval

FRENCH = """# Été à Marseille

L'été à Marseille est chaud et sec. Le mistral souffle depuis la vallée du
Rhône et rafraîchit la côte. Les calanques attirent des baigneurs, et la
bouillabaisse se mange sur le Vieux-Port, préparée avec des rascasses et du
safran. Les cigales chantent dans les pins pendant les après-midi brûlants.

# Fêtes de Noël en Provence

À Noël, les familles provençales dressent la crèche avec des santons
d'argile. Les treize desserts rappellent le Christ et les apôtres : nougat
blanc et noir, dattes, figues sèches, amandes et la pompe à l'huile. La
messe de minuit se célèbre dans les églises des villages perchés.
"""

GREEK = """# Ακρόπολη της Αθήνας

Η Ακρόπολη είναι ένας βράχος πάνω από την Αθήνα. Ο Παρθενώνας χτίστηκε
τον πέμπτο αιώνα προ Χριστού προς τιμήν της θεάς Αθηνάς. Το Ερέχθειο με τις
Καρυάτιδες βρίσκεται στη βόρεια πλευρά του βράχου, και τα Προπύλαια
αποτελούν τη μνημειώδη είσοδο.

# Ελαιόλαδο

Η Κρήτη παράγει εξαιρετικό ελαιόλαδο από την ποικιλία κορωνέικη. Οι ελιές
μαζεύονται τον χειμώνα και πιέζονται στα ελαιοτριβεία μέσα σε λίγες ώρες,
ώστε το λάδι να έχει χαμηλή οξύτητα και έντονο άρωμα.
"""

GERMAN = """# Straßenbahn in Zürich

Die Straßenbahn fährt in Zürich seit dem neunzehnten Jahrhundert. Heute
verbinden blaue Wagen die Altstadt mit den Quartieren am Seeufer, und die
Fahrgäste lösen ihre Billette an Automaten an jeder Haltestelle. Die ﬁnale
Haltestelle liegt oft weit draußen.
"""

# (question, expected section, expected source) — none of these can be
# answered by the old [a-z0-9'] tokenizer
QUESTIONS = [
    ("Pourquoi l'été est-il chaud à Marseille ?", "Été à Marseille",
     "provence.md"),
    ("Quels sont les treize desserts de Noël ?", "Fêtes de Noël en Provence",
     "provence.md"),
    ("Ποιος χτίστηκε ο Παρθενώνας στην Ακρόπολη;", "Ακρόπολη της Αθήνας",
     "hellas.md"),
    ("Πότε ΜΑΖΕΎΟΝΤΑΙ οι ελιές για το ελαιόλαδο;", "Ελαιόλαδο", "hellas.md"),
    # one '#' heading only → the whole file is one section named by stem
    ("STRASSENBAHN ZÜRICH", "zurich", "zurich.md"),
]
OFF_TOPIC = ["Comment réparer un vélo électrique ?",
             "Τι είναι η κβαντική φυσική;"]


def _old_tokenize(text):
    """Verbatim pre-P2-6 tokenizer (stopwords = the default list)."""
    return [rag._light_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in rag.STOPWORDS and len(w) > 1]


# -------------------------------------------------------------- tokenize --

def test_unicode_tokens():
    assert rag.tokenize("Crème brûlée à l'été") == ["crème", "brûlée", "l'été"]
    assert rag.tokenize("Straße") == rag.tokenize("STRASSE") == ["strasse"]
    assert rag.tokenize("ﬁnale Ｚürich") == ["finale", "zürich"]   # NFKC
    assert rag.tokenize("ΟΔΟΣ οδός") == ["οδοσ", "οδόσ"]     # final sigma
    # casefold keeps accents: an all-caps Greek word written without its
    # tonos does not match the accented form (no accent folding, P2-6)
    assert rag.tokenize("ΕΛΑΙΟΛΑΔΟ") != rag.tokenize("ελαιόλαδο")
    assert rag.tokenize("snake_case foo") == ["snake", "case", "foo"]
    # combining marks stay inside the word (Devanagari vowel signs/virama)
    assert rag.tokenize("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    # decomposed and precomposed accents are the same token
    assert rag.tokenize("café") == rag.tokenize("café")


def test_ascii_english_unchanged():
    """English (ASCII) text tokenizes exactly as before."""
    texts = [c["q"] for c in load_eval()]
    for f in ("beekeeping.md", "volcanoes.txt", "report.txt", "report.md",
              "tiny.txt"):
        texts.append((CORPUS / f).read_text())
    texts.append("It's the rocks' 3rd test -- O'Brien's ''quoted'' 42nd_st")
    for t in texts:
        assert rag.tokenize(t) == _old_tokenize(t)


# --------------------------------------------------- non-ASCII retrieval --

@pytest.fixture
def intl_col(tmp_path):
    d = tmp_path / "intl"
    d.mkdir()
    (d / "provence.md").write_text(FRENCH)
    (d / "hellas.md").write_text(GREEK)
    (d / "zurich.md").write_text(GERMAN)
    for f in CLEAN_FILES:               # English neighbours
        (d / f).write_bytes((CORPUS / f).read_bytes())
    col = library.create("Intl", link=str(d))
    ingest.run_pipeline(col, summaries=False, workers=1,
                        progress=lambda e: None)
    return library.get(col.id)


@pytest.mark.parametrize("q,section,source", QUESTIONS)
def test_non_ascii_questions_retrieve(intl_col, q, section, source):
    index = rag.load_index(intl_col)
    hits = index.bm25.search(q, 3) if hasattr(index, "bm25") \
        else index.search(q, 3)
    assert hits, q
    assert (hits[0][1]["essay"], hits[0][1]["source"]) == (section, source)
    assert rag.gate_ok(index, q)
    if hasattr(index, "bm25"):                 # the BM25 signal alone too
        assert rag.bm25_gate_ok(index.bm25, q, settings=index)


@pytest.mark.parametrize("q", OFF_TOPIC)
def test_non_ascii_off_topic_bm25_gated(intl_col, q):
    index = rag.load_index(intl_col)
    bm = index.bm25 if hasattr(index, "bm25") else index
    assert not rag.bm25_gate_ok(bm, q, settings=index)


@pytest.mark.xfail(strict=True, reason=(
    "known limit, not the tokenizer: the English bge-small encoder puts "
    "any Greek question at cosine ~0.73-0.77 to Greek chunks, above "
    "DENSE_SOLO_SIM (0.72), so the dense signal alone accepts it"))
def test_greek_off_topic_hybrid_gate(intl_col):
    index = rag.load_index(intl_col)
    if not hasattr(index, "dense"):
        pytest.skip("no dense signal")       # BM25 gate rejects it (above)
    assert not rag.gate_ok(index, OFF_TOPIC[1])


def test_english_still_answered_in_mixed_collection(intl_col):
    index = rag.load_index(intl_col)
    for q in load_eval():
        assert rag.gate_ok(index, q["q"]) == bool(q["expect"]), q["q"]


# ------------------------------------------------- per-collection stopwords --

def test_collection_stopwords_file(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    idx = rag.load_index(col)
    assert "magma" in idx.bm25.postings if hasattr(idx, "bm25") \
        else "magma" in idx.postings
    (col.root / "stopwords.txt").write_text(
        "# collection stopwords\nmagma\nVOLCANO\n-not\n\n")
    sw = rag.collection_stopwords(col)
    assert {"magma", "volcano"} <= sw and "not" not in sw
    assert "the" in sw                               # defaults kept
    idx = rag.load_index(library.get(col.id))
    bm = idx.bm25 if hasattr(idx, "bm25") else idx
    assert "magma" not in bm.postings and "volcano" not in bm.postings
    assert bm.tokenize("magma not volcano rock") == ["not", "rock"]
    # the token cache was rebuilt for the new list (tag differs)
    cache = next((col.data_dir / "cache" / "tokens").glob("*.json"))
    assert json.loads(cache.read_text())["tag"] == rag._tokenizer_tag(sw)
    assert rag._tokenizer_tag(sw) != rag._tokenizer_tag()
    # the BM25 gate uses the same list: a question made only of stopwords
    assert not rag.bm25_gate_ok(bm, "magma volcano", settings=idx)


def test_no_stopwords_file_means_defaults(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    assert rag.collection_stopwords(col) == frozenset(rag.STOPWORDS)


def test_tokenizer_version_bumped():
    assert rag.TOKENIZER_VERSION >= 2
