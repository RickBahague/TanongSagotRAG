# PyInstaller spec for the AskJ desktop app (one-folder build).
# Prereqs (run once on the build host):
#   .venv/bin/pip install -r requirements-app.txt
#   .venv/bin/python scripts/fetch_model.py
#   .venv/bin/python scripts/export_onnx.py
# Build:
#   .venv/bin/pyinstaller askj.spec
import sys
from pathlib import Path

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "static"), "static"),
    (str(ROOT / "data"), "data"),          # chunks + vectors (pre-ingested)
    (str(ROOT / "writings"), "writings"),  # bundled library (see rights note)
    (str(ROOT / "models"), "models"),      # GGUF + ONNX encoder
]

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["llama_cpp", "onnxruntime", "tokenizers", "webview"],
    excludes=["torch", "sentence_transformers", "transformers",
              "tkinter", "matplotlib"],
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, exclude_binaries=True, name="AskJ",
          console=False, icon=str(ROOT / "assets" / "askj.icns")
          if sys.platform == "darwin" and (ROOT / "assets" / "askj.icns").exists()
          else None)

coll = COLLECT(exe, a.binaries, a.datas, name="AskJ")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="AskJ.app", bundle_identifier="org.askj.app",
                 info_plist={"NSHighResolutionCapable": True,
                             "LSMinimumSystemVersion": "12.0"})
