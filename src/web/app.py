import asyncio
import websockets
import json
import uuid
from typing import Dict
import sys
import os

# Add the src directory to the path so we can import our game modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from __init__ import get_logger
from web.websocket_cli_bridge import WebSocketCLIBridge

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
            for player in bridge.game_state.players:
                client_id = bridge.get_client_id_for_player(player)
                if client_id:
                    await self.send_to_client(client_id, message)

    async def handle_create_game(self, client_id: str, data: dict):
        """Handle game creation request"""
        num_players = data.get('num_players', 2)
        player_name = data.get('player_name', f'Player_{client_id[:8]}')

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

        waiting_room = self.waiting_rooms[session_id]

        if len(waiting_room['players']) >= waiting_room['max_players']:
            await self.send_to_client(client_id, {
                'type': 'error',
                'message': 'Game session is full'
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
                player = cli_bridge.get_player_by_client_id(client_id)
                if player:
                    # Handle player disconnect in your existing GameState
                    await self.broadcast_to_session(session_id, {
                        'type': 'player_left',
                        'player_name': player.name
                    })

                    # Could implement reconnection logic here
                    # For now, end the game if someone leaves
                    del self.active_games[session_id]

            if client_id in self.client_to_session:
                del self.client_to_session[client_id]

    async def handle_message(self, websocket, client_id: str, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'create_game':
                await self.handle_create_game(client_id, data)
            elif msg_type == 'join_game':
                await self.handle_join_game(client_id, data)
            elif msg_type == 'player_action':
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

    async def client_handler(self, websocket, path):
        """Handle individual client connections"""
        client_id = str(uuid.uuid4())
        await self.register_client(websocket, client_id)

        try:
            async for message in websocket:
                await self.handle_message(websocket, client_id, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(client_id)

    def start_server(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting Regicide WebSocket server on {self.host}:{self.port}")

        start_server = websockets.serve(
            self.client_handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )

        return start_server

if __name__ == "__main__":
    server = RegicideWebSocketServer()
    start_server = server.start_server()

    print(f"Regicide WebSocket server starting on ws://{server.host}:{server.port}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()
