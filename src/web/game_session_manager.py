import uuid
from typing import Dict, List, Optional
from game.player import Player

class GameSession:
    def __init__(self, session_id: str, max_players: int):
        self.session_id = session_id
        self.max_players = max_players
        self.players: Dict[str, str] = {}  # client_id -> player_name
        self.game_state = None
        self.engine = None
        self.is_active = False

    def add_player(self, client_id: str, player_name: str) -> bool:
        """Add a player to the session"""
        if len(self.players) >= self.max_players:
            return False

        self.players[client_id] = player_name
        return True

    def remove_player(self, client_id: str) -> bool:
        """Remove a player from the session"""
        if client_id in self.players:
            del self.players[client_id]
            return True
        return False

    def get_player_name(self, client_id: str) -> Optional[str]:
        """Get player name by client ID"""
        return self.players.get(client_id)

    def get_player_client_ids(self) -> List[str]:
        """Get all client IDs in this session"""
        return list(self.players.keys())

    def get_player_count(self) -> int:
        """Get current number of players"""
        return len(self.players)

    def is_full(self) -> bool:
        """Check if session has maximum players"""
        return len(self.players) >= self.max_players

    def is_empty(self) -> bool:
        """Check if session has no players"""
        return len(self.players) == 0

    def create_game_players(self) -> List[Player]:
        """Create Player objects for the game engine"""
        hand_size_table = {2: 7, 3: 6, 4: 5}
        max_hand_size = hand_size_table[self.max_players]

        game_players = []
        for client_id, player_name in self.players.items():
            game_players.append(Player(name=player_name, hand_limit=max_hand_size))

        return game_players


class GameSessionManager:
    def __init__(self, logger):
        self.sessions: Dict[str, GameSession] = {}
        self.logger = logger

    def create_session(self, max_players: int) -> str:
        """Create a new game session"""
        session_id = str(uuid.uuid4())[:8]  # Short ID for easier sharing
        self.sessions[session_id] = GameSession(session_id, max_players)
        self.logger.info(f"Created game session {session_id} for {max_players} players")
        return session_id

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get a game session by ID"""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a game session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Removed game session {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.sessions.keys())

    def cleanup_empty_sessions(self):
        """Remove empty sessions"""
        empty_sessions = [sid for sid, session in self.sessions.items() if session.is_empty()]
        for session_id in empty_sessions:
            self.remove_session(session_id)
