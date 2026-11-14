#!/usr/bin/env python3
"""
Regicide WebSocket Server Launcher

This script starts the WebSocket server that allows remote players
to connect and play your CLI-based Regicide game.
"""

import asyncio
import sys
import os
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from web.app import RegicideWebSocketServer
from web.static_server import HTTP_PORT_OFFSET
from __init__ import get_logger


def main():
    parser = argparse.ArgumentParser(description='Start Regicide WebSocket Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to (default: 8765)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Set up logging
    logger = get_logger("regicide.server")
    logger.setLevel(args.log_level.upper())

    # Create and start server
    server = RegicideWebSocketServer(host=args.host, port=args.port)
    start_server = server.start_server()

    print("🎴 Regicide WebSocket Server")
    print("=" * 40)
    print(f"Server URL: ws://{args.host}:{args.port}")
    print(f"Static client: http://{args.host}:{args.port + HTTP_PORT_OFFSET}")
    print(f"Log Level: {args.log_level}")
    print("\nServer starting... Press Ctrl+C to stop")
    print("=" * 40)

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)
        loop.run_forever()
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
