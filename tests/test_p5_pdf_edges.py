"""Frequency-based running header/footer stripping for PDFs (JMS)."""
import ingest


def _pages(n=20):
    out = []
    for i in range(n):
        head = "Building Strength Through Struggle" if i % 2 else "Our Urgent Tasks"
        w = ["glacier", "harbor", "orchard", "quarry", "meadow", "lantern",
             "canyon", "violin", "saddle", "comet"]
        out.append(f"{head}\nBody about land reform and {w[i % 10]} {w[(i * 3) % 10]}.\n"
                   f"Then {w[(i * 7) % 10]} words, {w[(i + 4) % 10]} too.\n{i + 1}")
    return out


def test_repeated_headers_and_page_numbers_found():
    rep = ingest.repeated_edge_lines(_pages())
    assert "building strength through struggle" in rep
    assert "our urgent tasks" in rep
    assert "#" in rep                                   # bare page numbers
    assert not any("land reform" in r for r in rep)


def test_strip_keeps_body():
    pages = _pages()
    rep = ingest.repeated_edge_lines(pages)
    out = ingest.strip_edge_lines(pages[3], rep)
    assert out.splitlines() == pages[3].splitlines()[1:3]


def test_rare_lines_and_short_documents_untouched():
    pages = _pages(3)                         # below EDGE_MIN_PAGES
    assert ingest.repeated_edge_lines(pages) == set()
    assert ingest.strip_edge_lines(pages[0], set()) == pages[0]


def test_date_lines_and_list_numbers_kept():
    pages = [f"12 May 19{80 + i % 5}\nLetter body {i}.\n1." for i in range(20)]
    rep = ingest.repeated_edge_lines(pages)
    assert not any("may" in r for r in rep) and "#." not in rep


def test_body_line_mid_page_not_stripped():
    pages = _pages()
    rep = ingest.repeated_edge_lines(pages)
    page = "Intro line\nOur Urgent Tasks\nmore\nend\nlast"
    assert ingest.strip_edge_lines(page, rep) == page    # not at an edge


def test_off_by_default_and_params_unchanged(make_collection):
    import config
    col = make_collection(files=("volcanoes.txt",))
    assert config.PDF_STRIP_HEADERS == "off"
    assert "pdf_strip_headers" not in ingest.chunk_params(col)
    col.settings_file.write_text('{"PDF_STRIP_HEADERS": "on"}')
    import library
    assert ingest.chunk_params(library.get(col.id))["pdf_strip_headers"] is True


def test_page_texts_strip_only_when_asked():
    class P:
        def __init__(self, t):
            self.t = t

        def extract_text(self):
            return self.t

    class R:
        pages = [P(t) for t in _pages()]
    plain = ingest._pdf_page_texts(R())
    assert plain == _pages()
    stripped = ingest._pdf_page_texts(R(), True)
    assert not any("Urgent Tasks" in t for t in stripped)
