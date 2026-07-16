# PyInstaller spec for the AnyRAG desktop app (one-folder build).
# Prereqs (run once on the build host):
#   .venv/bin/pip install -r requirements-app.txt
#   .venv/bin/python scripts/fetch_model.py      # GGUF for the embedded LLM
#   .venv/bin/python scripts/export_onnx.py      # ONNX query encoder
# Build:
#   .venv/bin/pyinstaller anyrag.spec            # or: bash scripts/build_desktop.sh
#
# Unlike AskJ, NO corpus is bundled — the app ships empty; users create
# collections from their own folders (all data in the user-data dir).
import sys
from pathlib import Path

ROOT = Path(SPECPATH)

datas = [(str(ROOT / "static"), "static")]
if (ROOT / "models").is_dir():
    datas.append((str(ROOT / "models"), "models"))   # GGUF + ONNX encoder
else:
    print("WARNING: models/ missing — bundle will have no embedded LLM or "
          "ONNX encoder. Run scripts/fetch_model.py + scripts/export_onnx.py "
          "first, or ship an Ollama-only build.")

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["llama_cpp", "onnxruntime", "tokenizers", "webview",
                   "docx"],
    excludes=["torch", "sentence_transformers", "transformers",
              "tkinter", "matplotlib"],
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, exclude_binaries=True, name="AnyRAG",
          console=False, icon=str(ROOT / "assets" / "anyrag.icns")
          if sys.platform == "darwin" and (ROOT / "assets" / "anyrag.icns").exists()
          else None)

coll = COLLECT(exe, a.binaries, a.datas, name="AnyRAG")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="AnyRAG.app", bundle_identifier="org.anyrag.app",
                 info_plist={"NSHighResolutionCapable": True,
                             "LSMinimumSystemVersion": "12.0"})
