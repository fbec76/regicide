import asyncio
from typing import Dict, List, Optional, Callable

# Import your existing game modules
from game.engine import Engine, deal_initial_hands
from game.game_state import GameState
from game.deck import TavernDeck, CastleDeck, DiscardPile
from game.player import Player
from game.rules import is_valid_play, total_attack
from __init__ import get_logger

class WebSocketCLIBridge:
    """
    Bridge between WebSocket server and your existing CLI-based game engine.
    This directly uses your GameState class instead of creating redundant session management.
    """

    def __init__(self, session_id: str, client_players: Dict[str, str], broadcast_callback: Callable, logger):
        self.session_id = session_id
        self.client_players = client_players  # {client_id: player_name}
        self.broadcast = broadcast_callback
        self.logger = logger

        # Your existing game components - no duplication!
        self.engine = None
        self.game_state = None  # This is YOUR GameState class

        # Player input queue for async handling
        self.pending_actions: Dict[str, asyncio.Event] = {}
        self.player_responses: Dict[str, any] = {}

    async def initialize_game(self):
        """Initialize the game using your existing GameState and Engine"""
        try:
            # Create hand size mapping from your main.py
            hand_size_table = {2: 7, 3: 6, 4: 5}
            max_hand_size = hand_size_table[len(self.client_players)]

            # Create Player objects using your existing Player class
            players = []
            for client_id, player_name in self.client_players.items():
                players.append(Player(name=player_name, hand_limit=max_hand_size))

            # Initialize your existing game components exactly as in main.py
            tavern_deck = TavernDeck()
            castle_deck = CastleDeck()
            discard_pile = DiscardPile()

            # Create YOUR GameState - no duplication!
            self.game_state = GameState(
                players,
                tavern_deck,
                castle_deck,
                discard_pile
            )

            # Initialize YOUR Engine exactly as in main.py
            self.engine = Engine(self.logger, get_logger)

            # Deal initial hands using your existing function
            deal_initial_hands(self.game_state)

            # Send initial game state to all players
            await self.broadcast_game_state()

            # Start the game loop using your existing engine logic
            asyncio.create_task(self.run_game_loop())

            self.logger.info(f"Game initialized for session {self.session_id} using existing GameState")

        except Exception as e:
            self.logger.error(f"Error initializing game: {e}")
            await self.broadcast(self.session_id, {
                'type': 'error',
                'message': f'Failed to initialize game: {str(e)}'
            })

    async def run_game_loop(self):
        """
        Main game loop that adapts your existing Engine to work with WebSocket.
        Uses your GameState directly - no duplication of game logic!
        """
        try:
            # This follows the same pattern as your existing Engine.run() method
            while not self.game_state.check_loss() and self.game_state.castle_deck.cards:
                active_player = self.game_state.get_active_player()

                # Send current state to all players
                await self.broadcast_game_state()

                # Get player action through WebSocket (replaces CLI input)
                action = await self.get_player_action(active_player)

                if action['type'] == 'play_cards':
                    success = await self.handle_play_cards(active_player, action['card_indices'])
                    if not success:
                        continue  # Invalid play, ask again
                elif action['type'] == 'yield':
                    await self.handle_yield(active_player)

                # Use your existing GameState methods for win/loss checks
                if self.game_state.check_loss():
                    await self.broadcast(self.session_id, {
                        'type': 'game_ended',
                        'result': 'loss',
                        'message': 'The players have been defeated!'
                    })
                    break

                if self.game_state.is_game_won():
                    await self.broadcast(self.session_id, {
                        'type': 'game_ended',
                        'result': 'victory',
                        'message': 'Congratulations! You have defeated all enemies!'
                    })
                    break

        except Exception as e:
            self.logger.error(f"Error in game loop: {e}")
            await self.broadcast(self.session_id, {
                'type': 'error',
                'message': f'Game error: {str(e)}'
            })

    async def get_player_action(self, player: Player) -> dict:
        """Get player action via WebSocket (replaces CLI input)"""
        client_id = self.get_client_id_for_player(player)
        if not client_id:
            raise Exception(f"Client ID not found for player {player.name}")

        # Send action request to specific player
        await self.broadcast(self.session_id, {
            'type': 'action_required',
            'player_name': player.name,
            'active_player': client_id,
            'options': ['play_cards', 'yield']
        })

        # Wait for player response
        event = asyncio.Event()
        self.pending_actions[client_id] = event

        await event.wait()

        # Get the response
        response = self.player_responses.get(client_id)
        if client_id in self.player_responses:
            del self.player_responses[client_id]
        if client_id in self.pending_actions:
            del self.pending_actions[client_id]

        return response

    async def handle_player_action(self, client_id: str, action_data: dict):
        """Handle incoming player action from WebSocket"""
        if client_id in self.pending_actions:
            # Store the response and signal that we received it
            self.player_responses[client_id] = action_data
            self.pending_actions[client_id].set()

    async def handle_play_cards(self, player: Player, card_indices: List[int]) -> bool:
        """Handle card play using your existing game rules and logic"""
        try:
            # Validate indices
            if any(idx < 0 or idx >= len(player.hand) for idx in card_indices):
                await self.send_to_player(player, {
                    'type': 'error',
                    'message': 'Invalid card selection'
                })
                return False

            # Get selected cards
            selected_cards = [player.hand[idx] for idx in sorted(card_indices, reverse=True)]

            # Validate play using your existing rules
            if not is_valid_play(selected_cards):
                await self.send_to_player(player, {
                    'type': 'error',
                    'message': 'Invalid card combination'
                })
                return False

            # Execute the play using your existing Player methods
            if not player.play_cards(selected_cards):
                await self.send_to_player(player, {
                    'type': 'error',
                    'message': 'Could not play selected cards'
                })
                return False

            # Calculate attack using your existing rules
            attack_value = total_attack(selected_cards)

            # Apply damage using your existing GameState
            self.game_state.enemy_damage += attack_value

            # Broadcast the play
            await self.broadcast(self.session_id, {
                'type': 'cards_played',
                'player_name': player.name,
                'cards': [card.display_name for card in selected_cards],
                'total_attack': attack_value
            })

            # Check if enemy is defeated using your existing logic
            if self.game_state.enemy_damage >= self.game_state.current_enemy.health:
                # Enemy defeated - use your existing GameState methods
                self.game_state.discard_pile.add_cards([self.game_state.current_enemy])
                self.game_state.current_enemy = self.game_state.castle_deck.draw_one()
                self.game_state.enemy_damage = 0

                await self.broadcast(self.session_id, {
                    'type': 'enemy_defeated',
                    'message': f'Enemy defeated! Next enemy appears.'
                })

            # Advance turn using your existing GameState method
            self.game_state.advance_turn()
            return True

        except Exception as e:
            self.logger.error(f"Error handling play cards: {e}")
            await self.send_to_player(player, {
                'type': 'error',
                'message': f'Error playing cards: {str(e)}'
            })
            return False

    async def handle_yield(self, player: Player):
        """Handle yield using your existing GameState logic"""
        try:
            # Use your existing yield logic
            if self.game_state.can_yield(player):
                self.game_state.yields_in_a_row[player.name] = True
                player.has_yielded = True

                # Advance turn using your existing method
                self.game_state.advance_turn()

                await self.broadcast(self.session_id, {
                    'type': 'player_yielded',
                    'player_name': player.name
                })
            else:
                await self.send_to_player(player, {
                    'type': 'error',
                    'message': 'Cannot yield - other players have already yielded'
                })
        except Exception as e:
            self.logger.error(f"Error handling yield: {e}")

    async def broadcast_game_state(self):
        """Send current game state using your existing GameState"""
        state_data = {
            'type': 'game_state',
            'current_enemy': {
                'name': self.game_state.current_enemy.display_name,
                'health': self.game_state.current_enemy.health,
                'attack': self.game_state.current_enemy.attack,
                'damage_taken': self.game_state.enemy_damage
            },
            'active_player': self.game_state.get_active_player().name,
            'turn_count': self.game_state.total_turns,
            'players': []
        }

        # Add player-specific data using your existing Player objects
        for player in self.game_state.players:
            client_id = self.get_client_id_for_player(player)
            if client_id:
                player_data = {
                    'name': player.name,
                    'hand_size': len(player.hand),
                    'is_active': player == self.game_state.get_active_player(),
                    'hand': [
                        {
                            'display_name': card.display_name,
                            'suit': card.suit,
                            'rank': card.rank,
                            'value': card.value
                        } for card in player.hand
                    ]
                }
                state_data['players'].append(player_data)

        await self.broadcast(self.session_id, state_data)

    def get_client_id_for_player(self, player: Player) -> Optional[str]:
        """Get client ID for a player"""
        for client_id, player_name in self.client_players.items():
            if player_name == player.name:
                return client_id
        return None

    def get_player_by_client_id(self, client_id: str) -> Optional[Player]:
        """Get Player object by client ID"""
        player_name = self.client_players.get(client_id)
        if player_name:
            for player in self.game_state.players:
                if player.name == player_name:
                    return player
        return None

    async def send_to_player(self, player: Player, message: dict):
        """Send message to specific player"""
        client_id = self.get_client_id_for_player(player)
        if client_id:
            message['target_player'] = player.name
            await self.broadcast(self.session_id, message)
