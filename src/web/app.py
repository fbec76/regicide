import asyncio
import inspect
import websockets
import json
import uuid
from typing import Dict, Set
import sys
import os

# Add the src directory to the path so we can import our game modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from __init__ import get_logger
from web.static_server import start_static_server
from web.websocket_cli_bridge import WebSocketCLIBridge

# Lobby hardening (security): combined cap on waiting rooms + active games.
MAX_CONCURRENT_ROOMS = 8
# Strict lobby validation: Regicide supports 2-4 players (solo unsupported).
VALID_PLAYER_COUNTS = {2, 3, 4}
# Cross-origin defense: only the Phase-B static sidecar's origin may connect.
ALLOWED_ORIGINS = ['http://localhost:8766']
# Inbound frame cap (security): malformed/oversized frames are dropped.
MAX_MESSAGE_SIZE = 65536

class RegicideWebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.logger = get_logger("regicide.websocket")

        # Track connected clients and their game sessions
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_to_session: Dict[str, str] = {}
        self.waiting_rooms: Dict[str, dict] = {}  # session_id -> {players: {client_id: name}, max_players: int}

        # Active game sessions using your existing GameState
        self.active_games: Dict[str, WebSocketCLIBridge] = {}

    async def register_client(self, websocket, client_id: str):
        """Register a new client connection"""
        self.connected_clients[client_id] = websocket
        self.logger.info(f"Client {client_id} connected")

        # Send welcome message
        await self.send_to_client(client_id, {
            'type': 'welcome',
            'client_id': client_id,
            'message': 'Connected to Regicide server'
        })

    async def unregister_client(self, client_id: str):
        """Unregister a client connection"""
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]

        # Remove from waiting room or active game
        if client_id in self.client_to_session:
            session_id = self.client_to_session[client_id]
            await self.handle_leave_game(client_id, session_id)

        self.logger.info(f"Client {client_id} disconnected")

    async def send_to_client(self, client_id: str, message: dict):
        """Send message to a specific client"""
        if client_id in self.connected_clients:
            try:
                await self.connected_clients[client_id].send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                await self.unregister_client(client_id)

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all clients in a session"""
        # Check if it's a waiting room
        if session_id in self.waiting_rooms:
            for client_id in self.waiting_rooms[session_id]['players']:
                await self.send_to_client(client_id, message)

        # Check if it's an active game
        elif session_id in self.active_games:
            bridge = self.active_games[session_id]
            for client_id in bridge.client_by_id:
                await self.send_to_client(client_id, message)

    async def handle_create_game(self, client_id: str, data: dict):
        """Handle game creation request"""
        num_players = data.get('num_players')
        player_name = data.get('player_name', f'Player_{client_id[:8]}')

        # Boundary validation (security): strict lobby schemas, no state
        # change on any rejection.
        if not isinstance(num_players, int) or isinstance(num_players, bool) \
                or num_players not in VALID_PLAYER_COUNTS:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'num_players must be 2, 3, or 4'
            })
            return
        if not isinstance(player_name, str) or not player_name.strip():
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'player_name must be a non-empty string'
            })
            return
        # One room per client.
        if client_id in self.client_to_session:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'You are already in a game session'
            })
            return
        # Concurrent room cap (waiting rooms + active games).
        if len(self.waiting_rooms) + len(self.active_games) >= MAX_CONCURRENT_ROOMS:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Server is full; try again later'
            })
            return

        # Create new waiting room
        session_id = str(uuid.uuid4())[:8]  # Short ID for easier sharing
        self.waiting_rooms[session_id] = {
            'players': {client_id: player_name},
            'max_players': num_players
        }
        self.client_to_session[client_id] = session_id

        await self.send_to_client(client_id, {
            'type': 'game_created',
            'session_id': session_id,
            'player_name': player_name,
            'players_needed': num_players
        })

        # Broadcast the creator's own join so their client's waiting-room
        # list populates (same message shape as handle_join_game).
        await self.broadcast_to_session(session_id, {
            'type': 'player_joined',
            'player_name': player_name,
            'players_count': len(self.waiting_rooms[session_id]['players']),
            'players_needed': self.waiting_rooms[session_id]['max_players']
        })

        self.logger.info(f"Game {session_id} created by {client_id}")

    async def handle_join_game(self, client_id: str, data: dict):
        """Handle joining an existing game"""
        session_id = data.get('session_id')
        player_name = data.get('player_name', f'Player_{client_id[:8]}')

        # Check waiting room
        if session_id not in self.waiting_rooms:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Game session not found'
            })
            return

        if not isinstance(player_name, str) or not player_name.strip():
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'player_name must be a non-empty string'
            })
            return

        waiting_room = self.waiting_rooms[session_id]

        if len(waiting_room['players']) >= waiting_room['max_players']:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Game session is full'
            })
            return

        # One room per client.
        if client_id in self.client_to_session:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'You are already in a game session'
            })
            return

        # Reject duplicate player names (identity is client_id-keyed, but
        # duplicate display names would make the table unreadable).
        if player_name in waiting_room['players'].values():
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'That player name is already taken in this session'
            })
            return

        # Add player to waiting room
        waiting_room['players'][client_id] = player_name
        self.client_to_session[client_id] = session_id

        # Notify all players in waiting room
        await self.broadcast_to_session(session_id, {
            'type': 'player_joined',
            'player_name': player_name,
            'players_count': len(waiting_room['players']),
            'players_needed': waiting_room['max_players']
        })

        # Start game if waiting room is full
        if len(waiting_room['players']) >= waiting_room['max_players']:
            await self.start_game_session(session_id)

        self.logger.info(f"Player {player_name} joined game {session_id}")

    async def start_game_session(self, session_id: str):
        """Start the game using your existing GameState and Engine"""
        waiting_room = self.waiting_rooms[session_id]

        # Create CLI bridge that uses your existing GameState
        cli_bridge = WebSocketCLIBridge(
            session_id,
            waiting_room['players'],  # {client_id: player_name}
            self.broadcast_to_session,
            self.send_to_client,
            self.logger
        )

        # Initialize game using your existing components
        await cli_bridge.initialize_game()

        # Move from waiting room to active game
        self.active_games[session_id] = cli_bridge
        del self.waiting_rooms[session_id]

        await self.broadcast_to_session(session_id, {
            'type': 'game_started',
            'message': 'Game has started! Check your hand and prepare for battle.'
        })

        self.logger.info(f"Game {session_id} started with your existing GameState")

    async def handle_player_action(self, client_id: str, data: dict):
        """Handle player game actions"""
        if client_id not in self.client_to_session:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'You are not in a game session'
            })
            return

        session_id = self.client_to_session[client_id]
        cli_bridge = self.active_games.get(session_id)

        if not cli_bridge:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Game session not found'
            })
            return

        # Forward action to CLI bridge (which uses your GameState)
        await cli_bridge.handle_player_action(client_id, data)

    async def handle_leave_game(self, client_id: str, session_id: str = None):
        """Handle player leaving a game"""
        if not session_id:
            session_id = self.client_to_session.get(client_id)

        if session_id:
            # Remove from waiting room
            if session_id in self.waiting_rooms:
                waiting_room = self.waiting_rooms[session_id]
                if client_id in waiting_room['players']:
                    player_name = waiting_room['players'][client_id]
                    del waiting_room['players'][client_id]

                    # Notify other players
                    await self.broadcast_to_session(session_id, {
                        'type': 'player_left',
                        'player_name': player_name
                    })

                    # Clean up empty waiting room
                    if not waiting_room['players']:
                        del self.waiting_rooms[session_id]

            # Remove from active game
            elif session_id in self.active_games:
                cli_bridge = self.active_games[session_id]
                player = cli_bridge.client_by_id.get(client_id)
                if player:
                    # A game that already ended must not be re-announced as
                    # 'abandoned' — the terminal game_ended broadcast (e.g.
                    # player_defeated) would be overwritten otherwise.
                    if not cli_bridge.game_over:
                        # Handle player disconnect in your existing GameState
                        await self.broadcast_to_session(session_id, {
                            'type': 'player_left',
                            'player_name': player.name
                        })

                        # Minimum reconnection behavior (documented): a
                        # player leaving an active game ends it for everyone.
                        await self.broadcast_to_session(session_id, {
                            'type': 'game_ended',
                            'result': 'abandoned'
                        })

                    # Remove the whole session and clear every member's
                    # session mapping.
                    for member_id in cli_bridge.client_by_id:
                        self.client_to_session.pop(member_id, None)
                    del self.active_games[session_id]

            if client_id in self.client_to_session:
                del self.client_to_session[client_id]

    async def handle_message(self, websocket, client_id: str, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type') if isinstance(data, dict) else None

            if not isinstance(data, dict):
                await self.send_to_client(client_id, {
                    'type': 'error',
                    'message': 'Invalid message: expected a JSON object'
                })
            elif msg_type == 'create_game':
                await self.handle_create_game(client_id, data)
            elif msg_type == 'join_game':
                await self.handle_join_game(client_id, data)
            elif msg_type in ('play', 'defend', 'jester_choose', 'preview'):
                await self.handle_player_action(client_id, data)
            elif msg_type == 'leave_game':
                await self.handle_leave_game(client_id)
            else:
                await self.send_to_client(client_id, {
                    'type': 'error',
                    'message': f'Unknown message type: {msg_type}'
                })

        except json.JSONDecodeError:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Invalid JSON message'
            })
        except Exception as e:
            self.logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Internal server error'
            })

    async def client_handler(self, websocket, path=None):
        """Handle individual client connections.

        ``path`` defaults to None because the handler signature must fit both
        websockets versions: 12.x legacy serve passes (connection, path), the
        >= 13 asyncio implementation passes only the connection.
        """
        client_id = str(uuid.uuid4())
        await self.register_client(websocket, client_id)

        # Concurrent per-connection dispatch (A6): every inbound message is
        # handled in its own task, so a handler that awaits a client reply —
        # the Jester peek-then-inject choice (bridge D3) — cannot deadlock
        # this read loop. Safety: engine.step() is synchronous and the
        # bridge validates expected-client/phase with NO await between
        # validation and step, so task interleaving can neither double-step
        # nor reorder engine state; while a Jester choice is pending the
        # bridge rejects every non-jester_choose action (bridge D3/D4).
        pending_tasks: Set[asyncio.Task] = set()

        def _on_done(task: asyncio.Task) -> None:
            pending_tasks.discard(task)

        try:
            async for message in websocket:
                task = asyncio.create_task(
                    self.handle_message(websocket, client_id, message))
                pending_tasks.add(task)
                task.add_done_callback(_on_done)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Cancel in-flight handlers BEFORE teardown so a pending action
            # can never race session cleanup. A handler can only be paused
            # at an await (Jester wait or an outbound send) — never between
            # validation and the synchronous step.
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await self.unregister_client(client_id)

    def start_server(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting Regicide WebSocket server on {self.host}:{self.port}")

        # Static file sidecar (Decision D1): daemon thread serving
        # src/web/static/ on WS port + 1 (default 8766), same host, strict
        # CSP. Runs only while the process lives.
        start_static_server(self.host, self.port, self.logger)

        start_server = websockets.serve(
            self.client_handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            origins=ALLOWED_ORIGINS,   # websockets 12.0 legacy serve kwarg
            max_size=MAX_MESSAGE_SIZE
        )

        return start_server

if __name__ == "__main__":
    async def _serve_forever():
        ws_server = RegicideWebSocketServer().start_server()
        if inspect.isawaitable(ws_server):
            ws_server = await ws_server
        try:
            await asyncio.get_running_loop().create_future()
        finally:
            ws_server.close()
            await ws_server.wait_closed()

    print(f"Regicide WebSocket server starting on ws://localhost:8765")
    try:
        asyncio.run(_serve_forever())
    except KeyboardInterrupt:
        pass
