#!/usr/bin/env bash
# Build the AnyRAG desktop app on the current host (macOS/Linux).
# One command: bash scripts/build_desktop.sh
set -e
cd "$(dirname "$0")/.."

[ -d .venv ] || bash setup.sh
.venv/bin/pip install -q -r requirements-app.txt

# assets the bundle needs (no corpus — the app ships empty)
.venv/bin/python scripts/fetch_model.py
[ -f models/encoder-onnx/model.onnx ] || .venv/bin/python scripts/export_onnx.py

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
