"""
CLI Adapter for WebSocket Integration

This module adapts your existing CLI display and input functions to work
with the WebSocket server while maintaining all game logic in the CLI.
"""

import asyncio
from typing import List, Dict, Any, Optional
from game.player import Player
from game.card import Card
from game.game_state import GameState

class WebSocketCLIAdapter:
    """
    Adapts CLI functions to work with WebSocket communication.
    This allows your existing CLI-based game logic to work with remote players.
    """

    def __init__(self, broadcast_callback, session_id: str):
        self.broadcast = broadcast_callback
        self.session_id = session_id
        self.pending_inputs: Dict[str, asyncio.Event] = {}
        self.input_responses: Dict[str, Any] = {}

    # Adapted Display Functions (send to WebSocket instead of terminal)

    async def show_current_state(self, game_state: GameState):
        """Adapted from cli.display.show_current_state"""
        enemy = game_state.current_enemy
        active_player = game_state.get_active_player()

        await self.broadcast(self.session_id, {
            'type': 'display_state',
            'enemy': {
                'name': enemy.display_name,
                'health': enemy.health,
                'attack': enemy.attack,
                'damage_taken': game_state.enemy_damage,
                'remaining_health': enemy.health - game_state.enemy_damage
            },
            'active_player': active_player.name,
            'tavern_cards_left': len(game_state.tavern_deck.cards),
            'castle_cards_left': len(game_state.castle_deck.cards)
        })

    async def show_player_hand(self, player: Player, client_id: str = None):
        """Adapted from cli.display.show_player_hand"""
        hand_data = []
        for i, card in enumerate(player.hand):
            hand_data.append({
                'index': i,
                'display_name': card.display_name,
                'suit': card.suit,
                'rank': card.rank,
                'value': card.value
            })

        message = {
            'type': 'show_hand',
            'player_name': player.name,
            'hand': hand_data
        }

        if client_id:
            message['target_client'] = client_id

        await self.broadcast(self.session_id, message)

    async def show_played_cards(self, player: Player, cards: List[Card]):
        """Adapted from cli.display.show_played_cards"""
        await self.broadcast(self.session_id, {
            'type': 'cards_played',
            'player_name': player.name,
            'cards': [card.display_name for card in cards],
            'total_attack': sum(card.value for card in cards)
        })

    async def show_enemy_defeated(self, enemy_name: str):
        """Adapted from cli.display.show_enemy_defeated"""
        await self.broadcast(self.session_id, {
            'type': 'enemy_defeated',
            'enemy_name': enemy_name,
            'message': f'{enemy_name} has been defeated!'
        })

    async def show_new_enemy(self, enemy):
        """Adapted from cli.display.show_new_enemy"""
        await self.broadcast(self.session_id, {
            'type': 'new_enemy',
            'enemy': {
                'name': enemy.display_name,
                'health': enemy.health,
                'attack': enemy.attack
            },
            'message': f'A new enemy appears: {enemy.display_name}!'
        })

    async def show_win(self):
        """Adapted from cli.display.show_win"""
        await self.broadcast(self.session_id, {
            'type': 'game_ended',
            'result': 'victory',
            'message': 'Congratulations! You have defeated all enemies and won the game!'
        })

    async def show_loss(self, reason: str):
        """Adapted from cli.display.show_loss"""
        await self.broadcast(self.session_id, {
            'type': 'game_ended',
            'result': 'defeat',
            'message': f'Game Over: {reason}'
        })

    async def show_invalid_action(self, message: str, player: Player = None):
        """Adapted from cli.display.show_invalid_action"""
        msg = {
            'type': 'invalid_action',
            'message': message
        }

        if player:
            msg['target_player'] = player.name

        await self.broadcast(self.session_id, msg)

    async def show_damage_dealt(self, damage: int, enemy_name: str):
        """Adapted from cli.display.show_damage_dealt"""
        await self.broadcast(self.session_id, {
            'type': 'damage_dealt',
            'damage': damage,
            'enemy_name': enemy_name,
            'message': f'Dealt {damage} damage to {enemy_name}!'
        })

    # Adapted Input Functions (get input from WebSocket instead of terminal)

    async def prompt_card_indexes_to_play(self, player: Player, client_id: str) -> Optional[List[Card]]:
        """Adapted from cli.input_handler.prompt_card_indexes_to_play"""
        # Send hand to player
        await self.show_player_hand(player, client_id)

        # Request card selection
        await self.broadcast(self.session_id, {
            'type': 'input_required',
            'input_type': 'card_selection',
            'player_name': player.name,
            'target_client': client_id,
            'message': 'Select cards to play (or yield)',
            'options': {
                'can_play_cards': True,
                'can_yield': True,
                'max_cards': len(player.hand)
            }
        })

        # Wait for response
        response = await self.wait_for_input(client_id)

        if response.get('action') == 'yield':
            return None

        if response.get('action') == 'play_cards':
            card_indices = response.get('card_indices', [])
            if not card_indices:
                return []

            # Validate indices and return cards
            valid_cards = []
            for idx in card_indices:
                if 0 <= idx < len(player.hand):
                    valid_cards.append(player.hand[idx])

            return valid_cards

        return []

    async def prompt_card_indexes_to_discard(self, player: Player, client_id: str, needed_value: int) -> List[Card]:
        """Adapted from cli.input_handler.prompt_card_indexes_to_discard"""
        await self.show_player_hand(player, client_id)

        await self.broadcast(self.session_id, {
            'type': 'input_required',
            'input_type': 'discard_selection',
            'player_name': player.name,
            'target_client': client_id,
            'message': f'You must discard cards worth at least {needed_value} points to defend against the enemy attack!',
            'required_value': needed_value,
            'options': {
                'must_discard': True,
                'min_value': needed_value
            }
        })

        response = await self.wait_for_input(client_id)
        card_indices = response.get('card_indices', [])

        # Validate and return cards
        valid_cards = []
        for idx in card_indices:
            if 0 <= idx < len(player.hand):
                valid_cards.append(player.hand[idx])

        return valid_cards

    async def prompt_action_choice(self, player: Player, client_id: str) -> str:
        """Adapted from cli.input_handler.prompt_action_choice"""
        await self.broadcast(self.session_id, {
            'type': 'input_required',
            'input_type': 'action_choice',
            'player_name': player.name,
            'target_client': client_id,
            'message': f'{player.name}, choose your action:',
            'options': {
                'actions': ['play', 'yield']
            }
        })

        response = await self.wait_for_input(client_id)
        return response.get('action', 'yield')

    async def wait_for_input(self, client_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for input from a specific client"""
        event = asyncio.Event()
        self.pending_inputs[client_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            response = self.input_responses.get(client_id, {})

            # Clean up
            if client_id in self.input_responses:
                del self.input_responses[client_id]
            if client_id in self.pending_inputs:
                del self.pending_inputs[client_id]

            return response

        except asyncio.TimeoutError:
            # Clean up on timeout
            if client_id in self.pending_inputs:
                del self.pending_inputs[client_id]
            return {'action': 'timeout'}

    def receive_input(self, client_id: str, input_data: Dict[str, Any]):
        """Receive input from WebSocket client"""
        if client_id in self.pending_inputs:
            self.input_responses[client_id] = input_data
            self.pending_inputs[client_id].set()

    # Utility methods for game state

    async def broadcast_game_update(self, game_state: GameState, message: str = None):
        """Send a general game update"""
        update = {
            'type': 'game_update',
            'current_enemy': {
                'name': game_state.current_enemy.display_name,
                'health': game_state.current_enemy.health,
                'attack': game_state.current_enemy.attack,
                'damage_taken': game_state.enemy_damage,
                'remaining_health': game_state.current_enemy.health - game_state.enemy_damage
            },
            'active_player': game_state.get_active_player().name,
            'turn_count': game_state.total_turns,
            'tavern_cards_left': len(game_state.tavern_deck.cards),
            'castle_cards_left': len(game_state.castle_deck.cards)
        }

        if message:
            update['message'] = message
