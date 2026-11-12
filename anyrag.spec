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

# llama-cpp-python loads libllama/libggml* with ctypes from
# <package>/lib at runtime; PyInstaller can't see that, and a hiddenimport
# alone packs only the Python code ("Shared library with base name 'llama'
# not found" in the app). Bundle the libraries at the same relative path.
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries = collect_dynamic_libs("llama_cpp")
if not any(Path(src).name.startswith("libllama") or
           Path(src).name.startswith("llama") for src, _ in binaries):
    print("WARNING: llama_cpp native libraries not found — the embedded "
          "LLM backend will not work in this build.")

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
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
