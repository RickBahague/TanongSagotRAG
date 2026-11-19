# PyInstaller spec for the AnyRAG desktop app (one-folder build).
# Prereqs (run once on the build host):
#   .venv/bin/pip install -r requirements-app.txt
#   .venv/bin/python scripts/export_onnx.py      # ONNX query encoder
#   .venv/bin/python scripts/fetch_reranker.py   # ONNX cross-encoder
#   .venv/bin/python scripts/fetch_model.py      # GGUF — only bundled with
#                                                # ANYRAG_BUNDLE_MODEL=1
# By default the 1.1 GB GGUF is NOT bundled: the app downloads it on first
# use into the user-data dir (modeldl.py), so the build is ~1 GB smaller.
# Build:
#   .venv/bin/pyinstaller anyrag.spec            # or: bash scripts/build_desktop.sh
#
# Unlike AskJ, NO corpus is bundled — the app ships empty; users create
# collections from their own folders (all data in the user-data dir).
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH)
MODELS = ROOT / "models"
GGUF = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

datas = [(str(ROOT / "static"), "static")]
for sub, why in (("encoder-onnx", "no ONNX encoder (BM25-only search); run "
                  "scripts/export_onnx.py"),
                 ("reranker-onnx", "no reranker; run scripts/fetch_reranker.py")):
    if (MODELS / sub).is_dir():
        datas.append((str(MODELS / sub), f"models/{sub}"))
    else:
        print(f"WARNING: models/{sub} missing — {why}")
if os.environ.get("ANYRAG_BUNDLE_MODEL") == "1":
    if (MODELS / GGUF).is_file():
        datas.append((str(MODELS / GGUF), "models"))
    else:
        print("WARNING: ANYRAG_BUNDLE_MODEL=1 but models/" + GGUF + " is "
              "missing; run scripts/fetch_model.py")

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
                   "docx", "smoke"],
    excludes=["torch", "sentence_transformers", "transformers",
              "tkinter", "matplotlib"],
)
pyz = PYZ(a.pure)

ICNS = ROOT / "assets" / "anyrag.icns"            # scripts/make_icon.py
ICON = str(ICNS) if sys.platform == "darwin" and ICNS.exists() else None

exe = EXE(pyz, a.scripts, exclude_binaries=True, name="AnyRAG",
          console=False, icon=ICON)

coll = COLLECT(exe, a.binaries, a.datas, name="AnyRAG")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="AnyRAG.app", bundle_identifier="org.anyrag.app",
                 icon=ICON,
                 info_plist={"NSHighResolutionCapable": True,
                             "LSMinimumSystemVersion": "12.0"})
