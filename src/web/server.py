#!/usr/bin/env python3
"""
Regicide WebSocket Server Launcher

This script starts the WebSocket server that allows remote players
to connect and play your CLI-based Regicide game.
"""

import asyncio
import inspect
import signal
import sys
import os
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from web.app import RegicideWebSocketServer
from web.static_server import HTTP_PORT_OFFSET
from __init__ import get_logger


async def async_main(host='localhost', port=8765):
    """Bind the WebSocket server (plus static sidecar) and serve forever.

    Works with both websockets versions the launcher may run under:
    - 12.x legacy serve: called inside a running loop, returns a Serve
      awaitable that binds to the current loop when awaited.
    - >= 13 asyncio implementation: requires a running loop at call time and
      returns the Server synchronously (still awaitable for compatibility).
    ``inspect.isawaitable`` covers both return shapes.
    """
    server = RegicideWebSocketServer(host, port)
    ws_server = server.start_server()  # also starts the static sidecar thread
    if inspect.isawaitable(ws_server):
        ws_server = await ws_server
    try:
        await asyncio.get_running_loop().create_future()  # serve forever
    finally:
        ws_server.close()
        await ws_server.wait_closed()


def main():
    parser = argparse.ArgumentParser(description='Start Regicide WebSocket Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to (default: 8765)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Set up logging
    logger = get_logger("regicide.server")
    logger.setLevel(args.log_level.upper())

    print("🎴 Regicide WebSocket Server")
    print("=" * 40)
    print(f"Server URL: ws://{args.host}:{args.port}")
    print(f"Static client: http://{args.host}:{args.port + HTTP_PORT_OFFSET}")
    print(f"Log Level: {args.log_level}")
    print("\nServer starting... Press Ctrl+C to stop")
    print("=" * 40)

    # Route SIGTERM through the same clean-exit path as Ctrl+C; the static
    # sidecar is a daemon thread and dies with the process.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    try:
        asyncio.run(async_main(args.host, args.port))
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutting down...")
        logger.info("Server shutdown requested")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logger.error(f"Server error: {e}")
    finally:
        print("Goodbye!")


if __name__ == "__main__":
    main()
