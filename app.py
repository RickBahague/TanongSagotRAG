#!/usr/bin/env python3
"""AnyRAG desktop entry point.

Starts the HTTP server on a free localhost port in a background thread,
then opens a native window (pywebview). Falls back to the default browser
if pywebview isn't available. Window close shuts everything down.
"""
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
    httpd = serve.make_server(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{config.HOST}:{port}"

    try:
        import webview  # pywebview
        webview.create_window("AnyRAG — ask your documents", url,
                              width=1040, height=780, min_size=(680, 500))
        webview.start()          # blocks until window closes
    except ImportError:
        print(f"pywebview not installed — opening {url} in your browser "
              f"(Ctrl-C to quit)")
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
