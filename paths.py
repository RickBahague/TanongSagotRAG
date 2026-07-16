"""Path resolution for dev and frozen (PyInstaller) desktop builds.

Bundle dir = read-only resources shipped with the app (static UI, models).
User dir   = ALL mutable state: the collection registry, every collection's
sources + derived data, settings, conversations. Unlike AskJ there is no
bundle/user data split — the app ships empty and everything users create
lives in one place.

Dev default: <repo>/devdata (git-ignored). Override with ANYRAG_HOME.
"""
import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
APP_NAME = "AnyRAG"


def bundle_dir() -> Path:
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).parent


def user_data_dir() -> Path:
    env = os.environ.get("ANYRAG_HOME")
    if env:
        d = Path(env)
    elif not FROZEN:
        d = Path(__file__).parent / "devdata"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / APP_NAME
    elif os.name == "nt":
        d = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    else:
        d = Path(os.environ.get("XDG_DATA_HOME",
                                Path.home() / ".local" / "share")) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
