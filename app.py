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


def main():
    port = free_port()
    token = secrets.token_urlsafe(32)
    httpd = serve.make_server(port, token=token)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # always 127.0.0.1: the server's Host allow-list is 127.0.0.1/localhost
    url = f"http://127.0.0.1:{port}/?token={token}"

    try:
        import webview  # pywebview
        webview.create_window("AnyRAG — ask your documents", url,
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
