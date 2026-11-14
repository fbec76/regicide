"""Static file sidecar (plan Decision D1): stdlib ``http.server`` daemon
thread serving ``src/web/static/`` next to the WebSocket server.

Why a sidecar module (imported by ``app.py``, started from
``RegicideWebSocketServer.start_server``) rather than inline in app.py:
content is 100% static, zero new dependencies, a daemon thread cannot
interfere with the asyncio event loop, and ``server.py`` stays the single
entrypoint (one command starts WS + HTTP).

Security hardening (review amendment sec#4):
- binds to the SAME host argument as the WS server (default ``localhost``),
- handler constructed with an explicit ``directory`` (never cwd/os.chdir),
- directory listings disabled (``403 Forbidden`` — chosen over an empty
  page so a misconfigured path is loudly visible in the browser),
- fail-fast on bind errors such as EADDRINUSE (clear log + re-raise),
- strict CSP header on EVERY response (including errors).

CSP note (Phase B): ``style-src 'self'`` forbids inline ``style=""``
attributes and ``<style>`` blocks. If the browser client ends up needing
inline styles, relax style-src to ``'self' 'unsafe-inline'`` — never
default-src.
"""
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Served directory is derived from this file's location — independent of
# the process working directory.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Port derivation: WS port + 1 (default 8765 → 8766). ``ws_port == 0``
# (ephemeral, used by the integration tests) keeps the sidecar ephemeral so
# the unmodified test suite never binds fixed ports.
HTTP_PORT_OFFSET = 1

# ``connect-src`` lists the default WS port AND the actual one so custom
# ``--port`` starts still connect. Same-origin default-src covers the static
# assets themselves.
CSP_TEMPLATE = ("default-src 'self'; "
                "connect-src ws://localhost:8765 ws://localhost:{ws_port}; "
                "style-src 'self'")


class StaticFileHandler(SimpleHTTPRequestHandler):
    """Serves ONLY ``STATIC_DIR`` with a strict CSP header on every response.

    ``csp_header`` and ``logger`` are class attributes set by
    ``start_static_server`` before the server starts (one sidecar per
    process). ``.js``/``.mjs`` MIME types are forced because the macOS
    system mimetypes registry sometimes mis-registers them.
    """

    csp_header = ""
    logger = None

    # Force correct types for the client's assets regardless of the OS
    # mimetypes registry (entries here win over mimetypes.guess_type).
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def list_directory(self, path):
        """Directory listings disabled (security hardening): refuse with
        403 instead of rendering an index page."""
        self.send_error(HTTPStatus.FORBIDDEN, "Directory listing is disabled")
        return None

    def end_headers(self):
        self.send_header("Content-Security-Policy", self.csp_header)
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        # Route per-request access logs to debug instead of stderr noise.
        if self.logger is not None:
            self.logger.debug(f"static: {format % args}")


def start_static_server(host: str, ws_port: int, logger) -> None:
    """Start the static sidecar as a daemon thread (runs while the process
    lives). Fails fast on bind errors (e.g. EADDRINUSE): logs a clear
    message and re-raises so the misconfiguration is not silently ignored."""
    port = 0 if ws_port == 0 else ws_port + HTTP_PORT_OFFSET
    StaticFileHandler.csp_header = CSP_TEMPLATE.format(ws_port=ws_port)
    StaticFileHandler.logger = logger
    try:
        server = ThreadingHTTPServer((host, port), StaticFileHandler)
    except OSError as exc:
        logger.error(
            f"Static file server cannot bind {host}:{port} ({exc}); "
            "is another Regicide instance already running?")
        raise
    thread = threading.Thread(
        target=server.serve_forever, name="regicide-static-server", daemon=True)
    thread.start()
    shown_port = str(port) if port else "<ephemeral>"
    logger.info(
        f"Serving static client at http://{host}:{shown_port} "
        f"(directory: {STATIC_DIR})")
