#!/usr/bin/env python3
"""AnyRAG desktop entry point.

Starts the HTTP server on a free localhost port in a background thread,
then opens a native window (pywebview). Falls back to the default browser
if pywebview isn't available. Window close shuts everything down.

Each launch generates a random session token: the server requires it (as an
HttpOnly cookie, set when the window first loads /?token=…), so other local
processes and web pages can't drive the app's API.
"""
import secrets
import socket
import threading
import webbrowser

import config
import serve


def free_port():
    with socket.socket() as s:
        s.bind((config.HOST, 0))
        return s.getsockname()[1]


class DesktopApi:
    """Exposed to the page as window.pywebview.api (P3-3). pywebview only
    exposes public methods; the window reference stays private (leading
    underscore) so it is never walked or serialized into JS."""

    def __init__(self):
        self._window = None

    def pick_folder(self):
        """Native folder picker: the chosen folder's path, or None when the
        dialog was cancelled (or no window exists yet)."""
        if self._window is None:
            return None
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
        if kind is None:                         # pywebview < 5
            kind = webview.FOLDER_DIALOG
        try:
            picked = self._window.create_file_dialog(kind)
        except Exception:
            return None
        if not picked:
            return None
        if isinstance(picked, str):
            return picked
        return str(picked[0]) if len(picked) else None


def main():
    port = free_port()
    token = secrets.token_urlsafe(32)
    httpd = serve.make_server(port, token=token)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # always 127.0.0.1: the server's Host allow-list is 127.0.0.1/localhost
    url = f"http://127.0.0.1:{port}/?token={token}"

    try:
        import webview  # pywebview
        api = DesktopApi()
        api._window = webview.create_window(
            "AnyRAG — ask your documents", url, js_api=api,
            width=1040, height=780, min_size=(680, 500))
        webview.start()          # blocks until window closes
    except ImportError:
        print(f"pywebview not installed — opening the app in your browser "
              f"at http://127.0.0.1:{port} (Ctrl-C to quit)")
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # PyInstaller + chunk-stage process pool
    main()
