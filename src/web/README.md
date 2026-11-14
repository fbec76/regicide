# Regicide WebSocket Server

This WebSocket server allows remote players to connect and play your CLI-based Regicide game while keeping all game logic handled through the existing CLI system.

## Architecture Overview

The WebSocket server structure maintains your existing CLI-based game logic while enabling multiplayer functionality:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Clients   │◄───┤ WebSocket Server│◄───┤ CLI Game Engine │
│  (Players)      │    │   (app.py)      │    │   (existing)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │ Game Sessions   │
                       │   Manager       │
                       └─────────────────┘
```

## File Structure

```
src/web/
├── __init__.py                 # Package initialization
├── app.py                      # Main WebSocket server
├── server.py                   # Server launcher script
├── websocket_cli_bridge.py     # Bridges WebSocket ↔ CLI game engine
├── cli_adapter.py              # Adapts CLI functions for WebSocket
└── client_example.py           # Example client for testing
```

## Key Components

### 1. WebSocket Server (`app.py`)
- Main server handling client connections
- Routes messages to appropriate game sessions
- Manages client registration and cleanup

### 2. Game Sessions
- Managed directly by `app.py` (waiting rooms, session registry)
- One game-setup path: `websocket_cli_bridge.WebSocketCLIBridge` builds
  `Player` objects and calls `Engine.reset`

### 3. WebSocket CLI Bridge (`websocket_cli_bridge.py`)
- **Core component** that integrates with your existing game engine
- Translates CLI input/output to WebSocket messages
- Runs your existing game loop asynchronously
- Maintains all game logic in CLI system

### 4. CLI Adapter (`cli_adapter.py`)
- Adapts your existing `cli.display` and `cli.input_handler` functions
- Sends display output via WebSocket instead of terminal
- Receives player input via WebSocket instead of keyboard

## Installation & Setup

1. Install WebSocket dependency:
```bash
pip install websockets~=12.0
```

2. Start the server:
```bash
# From src/web directory
python server.py

# Or with custom settings
python server.py --host 0.0.0.0 --port 8765 --log-level DEBUG
```

## How It Works

### Game Flow
1. **Server Start**: WebSocket server starts and listens for connections
2. **Player Connection**: Players connect using WebSocket clients
3. **Game Creation**: First player creates a game session
4. **Player Joining**: Other players join using the session ID
5. **Game Start**: Once all players join, your existing `Engine` starts
6. **CLI Integration**: All game logic flows through your existing CLI system:
   - `cli.display.*` functions → WebSocket messages to players
   - `cli.input_handler.*` functions → WebSocket input requests
   - `game.engine.Engine` → Unchanged, runs your existing game loop

### Message Protocol

#### Client → Server Messages
```json
{
  "type": "create_game",
  "player_name": "Alice",
  "num_players": 2
}

{
  "type": "join_game", 
  "session_id": "abc123",
  "player_name": "Bob"
}

{
  "type": "player_action",
  "action": "play_cards",
  "card_indices": [0, 2, 4]
}
```

#### Server → Client Messages
```json
{
  "type": "game_state",
  "current_enemy": {
    "name": "Jack of Hearts",
    "health": 20,
    "damage_taken": 5
  },
  "active_player": "Alice",
  "players": [...]
}

{
  "type": "input_required",
  "input_type": "card_selection", 
  "player_name": "Alice",
  "message": "Select cards to play"
}
```

## Testing the Server

### Method 1: Using the Example Client
```bash
# Terminal 1: Start server
cd src/web
python server.py

# Terminal 2: Player 1
python client_example.py
> create Alice 2

# Terminal 3: Player 2  
python client_example.py
> join <session_id> Bob
```

### Method 2: Manual WebSocket Testing
```javascript
// Browser console or Node.js
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({
  type: 'create_game',
  player_name: 'TestPlayer',
  num_players: 2
}));
```

## Integration Benefits

✅ **Preserves Existing Code**: Your entire game logic stays in CLI  
✅ **Minimal Changes**: No modifications to `game/` modules needed  
✅ **Async Compatible**: WebSocket server runs alongside your engine  
✅ **Scalable**: Supports multiple concurrent game sessions  
✅ **Testable**: Can still run CLI version independently  

## Development Notes

- The `WebSocketCLIBridge` is the key integration point
- All your existing `cli.display.*` functions are adapted to send WebSocket messages
- All your existing `cli.input_handler.*` functions are adapted to receive WebSocket input
- Your `game.engine.Engine` runs unchanged in async context
- Game state is managed by your existing `GameState` class

## Next Steps

1. **Test the basic server**: Run `python server.py` and connect with the example client
2. **Customize messages**: Modify the CLI adapter to match your specific display needs
3. **Add authentication**: Implement player authentication if needed
4. **Web UI**: Create a web interface that connects to this WebSocket server
5. **Production deployment**: Use proper WebSocket hosting (e.g., with nginx proxy)

The server maintains complete compatibility with your existing CLI game while enabling remote multiplayer functionality!
