#!/usr/bin/env bash
# Build the AnyRAG desktop app on the current host (macOS/Linux).
# One command: bash scripts/build_desktop.sh
set -e
cd "$(dirname "$0")/.."

[ -d .venv ] || bash setup.sh
.venv/bin/pip install -q -r requirements-app.txt

# assets the bundle needs (no corpus — the app ships empty). The 1.1 GB
# GGUF is downloaded by the app on first use unless ANYRAG_BUNDLE_MODEL=1.
[ -f models/encoder-onnx/model.onnx ] || .venv/bin/python scripts/export_onnx.py
.venv/bin/python scripts/fetch_reranker.py
if [ "${ANYRAG_BUNDLE_MODEL:-0}" = "1" ]; then .venv/bin/python scripts/fetch_model.py; fi
[ -f assets/anyrag.icns ] || .venv/bin/python scripts/make_icon.py

.venv/bin/pyinstaller --noconfirm anyrag.spec

# The embedded LLM needs llama.cpp's native library inside the bundle
# (anyrag.spec collects it; a missing one only shows up at the first question)
if [ "$(uname)" = "Darwin" ]; then BUNDLE=dist/AnyRAG.app; else BUNDLE=dist/AnyRAG; fi
if ! find "$BUNDLE" -path "*llama_cpp/lib/*" \( -name "libllama*" -o -name "llama*.dll" \) | grep -q .; then
  echo "BUILD CHECK FAILED: llama_cpp/lib/libllama is missing from $BUNDLE" >&2
  exit 1
fi

echo
echo "Build complete:"
if [ "$(uname)" = "Darwin" ]; then
  echo "  dist/AnyRAG.app   (dmg: hdiutil create -volname AnyRAG -srcfolder dist/AnyRAG.app -ov -format UDZO dist/AnyRAG.dmg)"
  echo "  Notarize before public release."
else
  echo "  dist/AnyRAG/      (AppImage: use appimagetool on a dist/AnyRAG AppDir)"
fi
