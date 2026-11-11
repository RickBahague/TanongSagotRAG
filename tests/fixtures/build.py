#!/usr/bin/env python3
"""Build the test fixture corpus (tests/fixtures/corpus/).

Deterministic, stdlib + pypdf + python-docx only (both core deps):

  .venv/bin/python tests/fixtures/build.py [OUT_DIR]

Each document covers a clearly distinct topic so retrieval/gate tests can
assert which source answers which question:

  beekeeping.md     markdown, 3 '#'-headed sections
  volcanoes.txt     plain text, single section
  lighthouses.html  html (title + paragraphs), single section
  glaciers.docx     Word, 3 'Heading 1' sections
  astronomy.pdf     4-page PDF with an outline whose titles repeat
                    ("Telescopes" appears twice → duplicate-section test)
  tiny.txt          20 words (short-document test)
  report.md         sourdough baking   ┐ same stem, different suffix
  report.txt        bicycle maintenance┘ (stem-collision test)
"""
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "corpus"

# Verbatim sentence used by the quote-verification tests.
QUOTE = ("A healthy colony will raise a new queen within days when the old "
         "queen is lost, feeding chosen larvae a diet of royal jelly.")

BEES = {
    "Hive Structure": (
        "The honey bee hive is built from hexagonal wax cells arranged in "
        "parallel combs. Worker bees secrete wax from glands on their "
        "abdomen and shape it with their mandibles. Brood cells in the "
        "centre of the comb hold eggs and larvae, while the outer cells "
        "store pollen and honey. A beekeeper inspects the frames of a "
        "Langstroth hive by lifting them one at a time and checking the "
        "brood pattern for gaps that may signal disease. " + QUOTE + " "
        "Propolis, a resin gathered from tree buds, seals cracks in the "
        "hive and protects the colony against microbes."),
    "Honey Harvest": (
        "Honey is harvested in late summer when the supers are full and "
        "the cells are capped with wax. The beekeeper removes the frames, "
        "slices the wax cappings with a heated knife, and spins the frames "
        "in a centrifugal extractor. Raw honey is strained through a fine "
        "mesh to remove wax fragments before bottling. Enough honey must "
        "always be left in the hive so the colony can survive the winter, "
        "since bees cluster together and eat stored honey to generate heat "
        "during cold months when no nectar is available."),
    "Swarming": (
        "Swarming is how a honey bee colony reproduces. When the hive "
        "becomes crowded, the old queen leaves with roughly half of the "
        "workers and settles temporarily on a branch while scout bees "
        "search for a new nest site. Scouts perform waggle dances to "
        "advertise candidate cavities, and the swarm departs once a quorum "
        "of scouts agrees on one location. Beekeepers prevent swarming by "
        "giving the colony more room, splitting strong hives, and removing "
        "queen cells during the spring buildup."),
}

VOLCANOES = (
    "Volcanoes form where molten rock called magma rises through the "
    "crust. At subduction zones an oceanic plate sinks beneath another "
    "plate, and water released from the descending slab lowers the melting "
    "point of the mantle above it. The resulting magma is rich in silica, "
    "viscous, and full of dissolved gas, which is why stratovolcanoes such "
    "as Mount Fuji and Mount St. Helens erupt explosively. Shield volcanoes "
    "like Mauna Loa in Hawaii form over hotspots and erupt runny basaltic "
    "lava that flows for many kilometres. Pyroclastic flows, avalanches of "
    "hot ash and gas moving faster than a car, are the deadliest volcanic "
    "hazard. Volcanologists monitor ground deformation with satellite radar, "
    "measure sulfur dioxide emissions, and record swarms of small "
    "earthquakes that often precede an eruption by days or weeks.")

LIGHTHOUSE_PARAS = [
    "Lighthouses guided ships along dangerous coasts for centuries. The "
    "Pharos of Alexandria, built in the third century BC, was one of the "
    "Seven Wonders of the Ancient World and burned an open fire at its top.",
    "In 1822 Augustin Fresnel invented a lens made of concentric glass "
    "prisms that bent scattered light into a single powerful horizontal "
    "beam. The Fresnel lens let a lighthouse beam be seen more than twenty "
    "nautical miles out to sea and was installed in lighthouses worldwide.",
    "Each lighthouse flashes a unique pattern, its characteristic, so that "
    "sailors can identify it at night. Keepers once lived at the station, "
    "trimming wicks, winding the clockwork that rotated the lens, and "
    "polishing the brass. Today almost every lighthouse is automated and "
    "monitored remotely.",
]

GLACIERS = {
    "Glacier Formation": (
        "A glacier forms where more snow falls each winter than melts each "
        "summer. Over many years the snow compacts into granular firn and "
        "then into dense blue glacial ice. When the ice becomes thick enough "
        "it begins to flow under its own weight, deforming plastically at "
        "depth while the brittle surface cracks into crevasses. Glaciers "
        "advance and retreat as the balance between accumulation and "
        "ablation shifts with the climate."),
    "Glacial Landforms": (
        "Moving ice carves the landscape. Valley glaciers grind V-shaped "
        "river valleys into broad U-shaped troughs, and where they reach the "
        "sea they leave deep fjords like those of Norway. Rocks frozen into "
        "the base of the ice scratch striations into bedrock. When a glacier "
        "retreats it leaves behind moraines, ridges of unsorted debris, as "
        "well as drumlins and eskers shaped by meltwater streams."),
    "Ice Cores": (
        "Ice cores drilled from the Antarctic and Greenland ice sheets "
        "preserve tiny bubbles of ancient air. By measuring carbon dioxide "
        "and methane in these bubbles, and the ratio of oxygen isotopes in "
        "the ice itself, scientists reconstruct temperature and atmospheric "
        "composition stretching back eight hundred thousand years. Layers "
        "of volcanic ash in the cores act as time markers."),
}

ASTRONOMY_PAGES = [  # (outline title, page text) — "Telescopes" repeats
    ("Telescopes",
     "Refracting telescopes gather light with an objective lens while "
     "reflecting telescopes use a curved primary mirror. Isaac Newton built "
     "the first practical reflector in 1668 to avoid the chromatic "
     "aberration that blurred colours in early refractors. The aperture of "
     "a telescope, the diameter of its main lens or mirror, determines how "
     "much light it collects and how fine a detail it can resolve."),
    ("Comets",
     "Comets are icy bodies left over from the formation of the solar "
     "system. Most originate in the distant Oort cloud or the Kuiper belt. "
     "As a comet approaches the Sun its ices sublimate, releasing gas and "
     "dust that form a glowing coma and two tails, a curved dust tail and a "
     "straight ion tail that always points away from the Sun. Halley's "
     "comet returns roughly every seventy six years."),
    ("Telescopes",
     "Modern observatories correct atmospheric turbulence with adaptive "
     "optics: a deformable mirror changes shape hundreds of times per "
     "second guided by a laser star. Space telescopes such as Hubble and "
     "the James Webb Space Telescope avoid the atmosphere entirely, and "
     "Webb observes in infrared light to see the first galaxies that "
     "formed after the Big Bang."),
    ("Nebulae",
     "A nebula is a cloud of interstellar gas and dust. Emission nebulae "
     "glow red because ultraviolet light from young hot stars ionizes "
     "hydrogen. Dark nebulae block the light of stars behind them, and "
     "planetary nebulae are the shells of gas thrown off by dying "
     "sun-like stars. The Orion Nebula is a stellar nursery where new stars "
     "are still forming today."),
]

TINY = ("Green ferns reproduce with spores rather than seeds and thrive in damp "
        "shaded forests on every continent except Antarctica today.")

SOURDOUGH = """# Sourdough Starter

A sourdough starter is a culture of wild yeast and lactic acid bacteria
kept alive in a mixture of flour and water. Bakers feed the starter daily
by discarding part of it and adding fresh flour and water. A vigorous
starter doubles in volume within a few hours of feeding and smells
pleasantly sour.

# Baking the Loaf

The dough is mixed, left to ferment in bulk, then shaped and proofed in a
banneton basket overnight in the refrigerator. Baking inside a preheated
cast iron Dutch oven traps steam, which lets the crust expand before it
sets and produces the open crumb and blistered crust of a good sourdough.
"""

BICYCLE = (
    "Bicycle maintenance starts with the chain. A clean, lubricated chain "
    "shifts smoothly and lasts longer, so wipe it with a rag, apply a drop "
    "of chain lube to each roller, and wipe off the excess. Check tyre "
    "pressure weekly with a gauge, because under-inflated tyres cause "
    "pinch flats. Brake pads wear down and should be replaced when the "
    "grooves disappear. Adjust the derailleur limit screws so the chain "
    "cannot fall off the largest or smallest cog, and tighten the stem and "
    "seatpost bolts to the torque printed on them.")


# ------------------------------------------------------------------ writers

def write_md(out):
    body = "\n\n".join(f"# {t}\n\n{textwrap.fill(x, 76)}"
                       for t, x in BEES.items())
    (out / "beekeeping.md").write_text(body + "\n")


def write_html(out):
    paras = "\n".join(f"<p>{p}</p>" for p in LIGHTHOUSE_PARAS)
    (out / "lighthouses.html").write_text(
        "<!DOCTYPE html><html><head><title>A Short History of Lighthouses"
        "</title><style>p{margin:0}</style><script>var x=1;</script></head>"
        f"<body><h1>Lighthouses</h1>\n{paras}\n</body></html>\n")


def write_docx(out):
    import docx
    d = docx.Document()
    import datetime
    d.core_properties.created = datetime.datetime(2026, 1, 1)
    for title, text in GLACIERS.items():
        d.add_heading(title, level=1)
        d.add_paragraph(text)
    d.save(str(out / "glaciers.docx"))


def _pdf_escape(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(out):
    """Hand-built text PDF (Helvetica, one page per entry), then pypdf adds
    the outline. pypdf extracts the text back line by line."""
    objs = []                                   # 1-based object bodies

    def add(body):
        objs.append(body)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = len(objs) + 1 + 2 * len(ASTRONOMY_PAGES)   # reserved below
    page_ids = []
    for title, text in ASTRONOMY_PAGES:
        lines = textwrap.wrap(text, 80)
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for ln in lines:
            ops.append(f"({_pdf_escape(ln)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream)
                      + stream + b"\nendstream")
        page_ids.append(add(
            (f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792]"
             f" /Resources << /Font << /F1 {font} 0 R >> >>"
             f" /Contents {content} 0 R >>").encode()))
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    assert add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
               .encode()) == pages_id
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(buf))
        buf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(buf)
    buf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        buf += b"%010d 00000 n \n" % off
    buf += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, catalog, xref))

    import io
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(clone_from=PdfReader(io.BytesIO(bytes(buf))))
    for i, (title, _) in enumerate(ASTRONOMY_PAGES):
        w.add_outline_item(title, i)
    w.add_metadata({"/Title": "Notes on Astronomy"})
    with open(out / "astronomy.pdf", "wb") as f:
        w.write(f)


def build(out=DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_md(out)
    (out / "volcanoes.txt").write_text(textwrap.fill(VOLCANOES, 76) + "\n")
    write_html(out)
    write_docx(out)
    write_pdf(out)
    (out / "tiny.txt").write_text(TINY + "\n")
    (out / "report.md").write_text(SOURDOUGH)
    (out / "report.txt").write_text(textwrap.fill(BICYCLE, 76) + "\n")
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT))
