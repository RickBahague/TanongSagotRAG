"""Desktop bundle bug (2026-11-13): llama-cpp-python was packed without its
native library, so `import llama_cpp` raised FileNotFoundError ("Shared
library with base name 'llama' not found"). The loader showed that raw text
and /backend/status still reported the embedded backend as reachable."""
import builtins

import pytest

import config
import rag


@pytest.fixture
def broken_llama(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "llama_cpp" or name.startswith("llama_cpp."):
            raise FileNotFoundError(
                "Shared library with base name 'llama' not found")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.setattr(rag, "_EMBEDDED_LLM", None)
    monkeypatch.setattr(rag, "_EMBEDDED_KEY", None)


def test_missing_native_library_message(broken_llama):
    msg = rag.llama_cpp_error()
    assert "built-in model could not be loaded" in msg and "Settings" in msg


def test_embedded_load_raises_clear_error(broken_llama, monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(config, "LLM_MODEL_PATH", gguf)
    with pytest.raises(RuntimeError, match="built-in model could not be loaded"):
        rag._embedded_llm()


def test_backend_status_not_reachable_without_native_lib(
        broken_llama, server, monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(config, "LLM_BACKEND", "embedded")
    monkeypatch.setattr(config, "LLM_MODEL_PATH", gguf)
    st, s = server.json("GET", "/backend/status")
    assert st == 200 and s["file_exists"] is True
    assert s["llama_cpp"] is False and s["reachable"] is False
    assert "built-in model could not be loaded" in s["error"]


def test_spec_bundles_llama_native_libs():
    from pathlib import Path
    spec = (Path(__file__).resolve().parent.parent / "anyrag.spec").read_text()
    assert 'collect_dynamic_libs("llama_cpp")' in spec
    assert "binaries=binaries" in spec
