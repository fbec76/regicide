"""Launcher-pattern test for ``src/web/server.py``.

Pins the version-agnostic launcher: ``async_main`` binds the WebSocket
server (under the project venv's websockets 12.x legacy serve, and by the
same ``isawaitable`` pattern under websockets >= 13 asyncio servers),
serves the app protocol over the wire, and exits cleanly on cancellation —
the code path ``main()`` takes on Ctrl+C.

The full serving behavior is covered by ``test_integration.py``; this test
only pins the launcher pattern itself.
"""
import asyncio
import json
import socket

import pytest
import websockets

from web.server import async_main  # noqa: E402  (conftest adds src to sys.path)

# The single origin app.py's cross-origin defense allows (A4 security).
ALLOWED_ORIGIN = "http://localhost:8766"
CONNECT_TIMEOUT = 5.0


def _free_port() -> int:
    """Reserve an ephemeral TCP port for the launcher to bind."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_async_main_starts_and_serves() -> None:
    """should bind the server, serve the welcome message, and cancel cleanly"""
    port = _free_port()

    async def scenario() -> None:
        # Arrange: launch the real launcher entrypoint as a task
        serve_task = asyncio.create_task(async_main("127.0.0.1", port))

        # Act: retry-connect until the launcher is accepting connections
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CONNECT_TIMEOUT
        ws = None
        while ws is None:
            assert loop.time() < deadline, \
                "async_main never started accepting connections"
            try:
                ws = await websockets.connect(
                    f"ws://127.0.0.1:{port}", origin=ALLOWED_ORIGIN)
            except OSError:
                await asyncio.sleep(0.05)

        # Assert: the server is live and speaks the app protocol
        try:
            message = json.loads(await asyncio.wait_for(
                ws.recv(), timeout=CONNECT_TIMEOUT))
            assert message["type"] == "welcome"
        finally:
            await ws.close()

        # Assert: cancellation (the Ctrl+C path) shuts the launcher down
        serve_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serve_task

    asyncio.run(asyncio.wait_for(scenario(), timeout=CONNECT_TIMEOUT * 3))
