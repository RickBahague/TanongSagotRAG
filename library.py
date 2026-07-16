"""Collection registry + the Collection class every pipeline stage runs
against. A collection = a set of source documents + their derived index
data + settings, fully isolated from other collections.

Layout (all under the user-data dir — see paths.py):

  collections.json                      registry: [{id, name, created}]
  collections/<id>/collection.json      name, description, author, persona,
                                        link (optional external folder),
                                        source_meta {filename: {…}}
  collections/<id>/sources/             managed documents (imports/copies)
  collections/<id>/data/chunks/         stage 1   one .jsonl per source
  collections/<id>/data/paras/          stage 1   display-only sidecar
  collections/<id>/data/dates/          stage 1c  sidecar
  collections/<id>/data/quotes/         stage 1d  sidecar
  collections/<id>/data/summaries/      stage 1b  output
  collections/<id>/data/vectors/        stage 2   output
  collections/<id>/data/conversations/  saved Q/A history
  collections/<id>/data/manifest.json   pipeline state per source
  collections/<id>/settings.json        per-collection overrides (optional)

Source modes: "managed" (files live in sources/) or "linked" (collection.json
has "link": an absolute path to the user's own folder; derived data still
lives under the collection, so the linked folder is never written to).
"""
import json
import re
import time
from pathlib import Path

import config

REGISTRY = config.USER_ROOT / "collections.json"
COLLECTIONS_DIR = config.USER_ROOT / "collections"

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".docx", ".html", ".htm")

_YEARS_RX = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")


class Collection:
    def __init__(self, cid, meta):
        self.id = cid
        self.meta = meta                      # contents of collection.json
        self.root = COLLECTIONS_DIR / cid

    # ------------------------------------------------------------- identity
    @property
    def name(self):
        return self.meta.get("name") or self.id

    @property
    def description(self):
        return self.meta.get("description", "")

    @property
    def author(self):
        """Single author of the corpus, when there is one ('' otherwise)."""
        return self.meta.get("author", "")

    @property
    def author_label(self):
        return self.author or "the author"

    @property
    def persona(self):
        """{'enabled': bool, 'model': '', 'backend': '', 'url': ''} or None.
        Persona mode is offered only when enabled AND an author is named."""
        p = self.meta.get("persona") or {}
        if p.get("enabled") and self.author:
            return p
        return None

    # ---------------------------------------------------------------- paths
    @property
    def sources_dir(self):
        link = self.meta.get("link")
        return Path(link) if link else self.root / "sources"

    @property
    def linked(self):
        return bool(self.meta.get("link"))

    @property
    def data_dir(self):
        return self.root / "data"

    @property
    def chunks_dir(self):
        return self.data_dir / "chunks"

    @property
    def paras_dir(self):
        return self.data_dir / "paras"

    @property
    def dates_dir(self):
        return self.data_dir / "dates"

    @property
    def quotes_dir(self):
        return self.data_dir / "quotes"

    @property
    def summaries_dir(self):
        return self.data_dir / "summaries"

    @property
    def vectors_dir(self):
        return self.data_dir / "vectors"

    @property
    def conversations_dir(self):
        return self.data_dir / "conversations"

    @property
    def manifest_file(self):
        return self.data_dir / "manifest.json"

    @property
    def settings_file(self):
        return self.root / "settings.json"

    # -------------------------------------------------------------- sources
    def source_files(self):
        """Supported documents in this collection, sorted by name."""
        d = self.sources_dir
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir()
                      if p.suffix.lower() in SUPPORTED_SUFFIXES
                      and not p.name.startswith("."))

    @property
    def source_meta(self):
        return self.meta.setdefault("source_meta", {})

    def work_title(self, filename):
        """Display title for a source file."""
        m = self.source_meta.get(Path(filename).name)
        if m and m.get("title"):
            return m["title"]
        return Path(filename).stem

    def work_years(self, filename):
        """(lo, hi) plausible year range for a source, or None. From explicit
        metadata, else years found in the title/filename (± slack)."""
        m = self.source_meta.get(Path(filename).name) or {}
        if m.get("years"):
            return tuple(m["years"])
        found = [int(y) for y in _YEARS_RX.findall(
            self.work_title(filename) + " " + Path(filename).name)]
        if found:
            return (min(found) - 3, max(found) + 3)
        return None

    def set_source_meta(self, filename, **kv):
        entry = self.source_meta.setdefault(Path(filename).name, {})
        entry.update({k: v for k, v in kv.items() if v is not None})
        self.save()

    # ------------------------------------------------------------- settings
    def setting(self, key, default=None):
        """Per-collection override, else global config value, else default."""
        s = self._settings()
        if key in s:
            return s[key]
        return getattr(config, key, default)

    def _settings(self):
        if not hasattr(self, "_settings_cache"):
            try:
                self._settings_cache = json.loads(self.settings_file.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                self._settings_cache = {}
        return self._settings_cache

    # -------------------------------------------------------------- persist
    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "collection.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2))

    def ensure_dirs(self):
        for d in (self.chunks_dir, self.paras_dir, self.dates_dir,
                  self.quotes_dir, self.summaries_dir, self.vectors_dir,
                  self.conversations_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.linked:
            self.sources_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ registry

def _slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "collection"
    existing = {e["id"] for e in _registry()}
    if s not in existing:
        return s
    n = 2
    while f"{s}-{n}" in existing:
        n += 1
    return f"{s}-{n}"


def _registry():
    try:
        return json.loads(REGISTRY.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_registry(entries):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def list_all():
    """[Collection] in registry order."""
    out = []
    for e in _registry():
        try:
            out.append(get(e["id"]))
        except KeyError:
            continue
    return out


def get(cid):
    cid = re.sub(r"[^\w-]", "", cid)
    f = COLLECTIONS_DIR / cid / "collection.json"
    if not f.exists():
        raise KeyError(cid)
    return Collection(cid, json.loads(f.read_text()))


def create(name, link=None, description="", author="", persona=None):
    """Create a collection; returns it. `link` = absolute path to an existing
    folder of documents (linked mode); omit for managed (imports go to
    sources/)."""
    if link:
        link = str(Path(link).expanduser().resolve())
        if not Path(link).is_dir():
            raise ValueError(f"Not a folder: {link}")
    cid = _slug(name)
    meta = {"name": name, "description": description, "author": author,
            "created": time.strftime("%Y-%m-%d")}
    if link:
        meta["link"] = link
    if persona:
        meta["persona"] = persona
    col = Collection(cid, meta)
    col.save()
    col.ensure_dirs()
    _save_registry(_registry() + [{"id": cid, "name": name,
                                   "created": meta["created"]}])
    return col


def delete(cid):
    """Remove a collection and all its derived data. A linked source folder
    is never touched."""
    import shutil
    col = get(cid)                      # validates id
    shutil.rmtree(col.root, ignore_errors=True)
    _save_registry([e for e in _registry() if e["id"] != cid])
