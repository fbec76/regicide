import unittest
from unittest.mock import patch, MagicMock
from collections import deque
from src.game.game_state import GameState
from src.game.player import Player
from src.game.card import Card
from src.game.enemy import Enemy
from src.game.deck import Deck, DiscardPile

class TestGameState(unittest.TestCase):
    def setUp(self):
        # Create players
        self.player1 = Player("Player1", 7)
        self.player2 = Player("Player2", 7)
        self.players = [self.player1, self.player2]
        
        # Create enemy cards for castle deck
        self.jack = Enemy("hearts", "J", 10, "enemy", "Jack of Hearts", 20, 10)
        self.queen = Enemy("diamonds", "Q", 15, "enemy", "Queen of Diamonds", 30, 15)
        
        # Create decks
        self.tavern_deck = Deck([
            Card("hearts", "5", 5, "number", "5 of Hearts"),
            Card("diamonds", "6", 6, "number", "6 of Diamonds")
        ])
        self.castle_deck = Deck([self.jack, self.queen])
        self.discard_pile = DiscardPile([])
        
        # Create game state
        self.game_state = GameState(self.players, self.tavern_deck, self.castle_deck, self.discard_pile)
        
    def test_initialization(self):
        """Test game state initialization"""
        # Check players
        self.assertEqual(len(self.game_state.players), 2)
        self.assertEqual(self.game_state.players[0], self.player1)
        self.assertEqual(self.game_state.players[1], self.player2)
        
        # Check decks
        self.assertEqual(self.game_state.tavern_deck, self.tavern_deck)
        self.assertEqual(self.game_state.castle_deck, self.castle_deck)
        self.assertEqual(self.game_state.discard_pile, self.discard_pile)
        
        # Check current enemy (should be the jack, drawn from castle deck)
        self.assertEqual(self.game_state.current_enemy, self.jack)
        self.assertEqual(len(self.castle_deck.cards), 1)  # Queen should still be in castle deck
        
        # Check initial state values
        self.assertEqual(self.game_state.enemy_damage, 0)
        self.assertEqual(self.game_state.spade_shield, 0)
        self.assertTrue(self.game_state.enemy_immunity)
        self.assertFalse(self.game_state.face_jester_played)
        self.assertEqual(self.game_state.played_this_enemy, [])
        
        # Check yield tracking
        self.assertEqual(self.game_state.yields_in_a_row, {"Player1": False, "Player2": False})
        
    def test_start_turn(self):
        """Test starting a new turn"""
        # Set up some state to reset
        self.game_state.spade_shield = 5
        self.game_state.enemy_damage = 10
        self.game_state.face_jester_played = True
        self.game_state.played_this_enemy = [Card("hearts", "5", 5, "number", "5 of Hearts")]
        
        # Start a new turn
        self.game_state.start_turn()
        
        # Check that state was reset
        self.assertEqual(self.game_state.spade_shield, 0)
        self.assertEqual(self.game_state.enemy_damage, 0)
        self.assertFalse(self.game_state.face_jester_played)
        self.assertEqual(self.game_state.played_this_enemy, [])
        
    def test_get_active_player(self):
        """Test getting the active player"""
        self.assertEqual(self.game_state.get_active_player(), self.player1)
        
    def test_advance_turn(self):
        """Test advancing to the next player's turn"""
        self.assertEqual(self.game_state.get_active_player(), self.player1)
        
        # Advance turn
        self.game_state.advance_turn()
        
        # Check that player2 is now active
        self.assertEqual(self.game_state.get_active_player(), self.player2)
        
        # Advance turn again
        self.game_state.advance_turn()
        
        # Check that player1 is active again (rotation)
        self.assertEqual(self.game_state.get_active_player(), self.player1)
        
    def test_defeat_enemy_discard(self):
        """Test defeating an enemy and discarding it"""
        # Initial state
        self.assertEqual(self.game_state.current_enemy, self.jack)
        self.assertEqual(len(self.discard_pile.cards), 0)
        
        # Add some cards to played_this_enemy
        card1 = Card("hearts", "5", 5, "number", "5 of Hearts")
        card2 = Card("diamonds", "6", 6, "number", "6 of Diamonds")
        self.game_state.played_this_enemy = [card1, card2]
        
        # Defeat enemy (not exact)
        self.game_state.defeat_enemy(exact=False)
        
        # Check that jack was added to discard pile
        self.assertEqual(len(self.discard_pile.cards), 3)  # Jack + 2 played cards
        self.assertIn(self.jack, self.discard_pile.cards)
        self.assertIn(card1, self.discard_pile.cards)
        self.assertIn(card2, self.discard_pile.cards)
        
        # Check that queen is now the current enemy
        self.assertEqual(self.game_state.current_enemy, self.queen)
        self.assertEqual(len(self.castle_deck.cards), 0)  # Castle deck should be empty
        
        # Check that state was reset
        self.assertEqual(self.game_state.enemy_damage, 0)
        self.assertEqual(self.game_state.spade_shield, 0)
        self.assertFalse(self.game_state.face_jester_played)
        self.assertEqual(self.game_state.played_this_enemy, [])
        
    def test_defeat_enemy_exact(self):
        """Test defeating an enemy with exact damage"""
        # Initial state
        self.assertEqual(self.game_state.current_enemy, self.jack)
        
        # Add some cards to played_this_enemy
        card1 = Card("hearts", "5", 5, "number", "5 of Hearts")
        card2 = Card("diamonds", "6", 6, "number", "6 of Diamonds")
        self.game_state.played_this_enemy = [card1, card2]
        
        # Defeat enemy (exact)
        self.game_state.defeat_enemy(exact=True)
        
        # Check that jack was added to top of tavern deck
        self.assertEqual(len(self.tavern_deck.cards), 3)  # Original 2 + Jack
        self.assertEqual(self.tavern_deck.cards[0], self.jack)  # Jack should be on top
        
        # Check that played cards were added to discard pile
        self.assertEqual(len(self.discard_pile.cards), 2)
        self.assertIn(card1, self.discard_pile.cards)
        self.assertIn(card2, self.discard_pile.cards)
        
    def test_update_enemy_immunity(self):
        """Test updating enemy immunity based on jester play"""
        # Initially immunity is on
        self.assertTrue(self.game_state.enemy_immunity)
        
        # Play a jester
        self.game_state.face_jester_played = True
        self.game_state.update_enemy_immunity()
        
        # Immunity should be off
        self.assertFalse(self.game_state.enemy_immunity)
        
        # Reset jester play
        self.game_state.face_jester_played = False
        self.game_state.update_enemy_immunity()
        
        # Immunity should be back on
        self.assertTrue(self.game_state.enemy_immunity)
        
    def test_play_card_against_enemy(self):
        """Test playing cards against the enemy"""
        # Create cards to play
        card1 = Card("hearts", "5", 5, "number", "5 of Hearts")
        card2 = Card("diamonds", "6", 6, "number", "6 of Diamonds")
        
        # Add cards to player's hand
        self.player1.draw_cards([card1, card2])
        
        # Play cards against enemy
        self.game_state.play_card_against_enemy(self.player1, [card1, card2], None)
        
        # Check that cards were removed from player's hand
        self.assertEqual(len(self.player1.hand), 0)
        
        # Check that cards were added to played_this_enemy
        self.assertEqual(len(self.game_state.played_this_enemy), 2)
        self.assertIn(card1, self.game_state.played_this_enemy)
        self.assertIn(card2, self.game_state.played_this_enemy)
        
    def test_apply_spades_shield(self):
        """Test applying a spades shield"""
        self.assertEqual(self.game_state.spade_shield, 0)
        
        # Apply shield
        self.game_state.apply_spades_shield(5)
        self.assertEqual(self.game_state.spade_shield, 5)
        
        # Apply more shield
        self.game_state.apply_spades_shield(3)
        self.assertEqual(self.game_state.spade_shield, 8)
        
    def test_apply_damage_to_enemy(self):
        """Test applying damage to the enemy"""
        self.assertEqual(self.game_state.enemy_damage, 0)
        
        # Apply damage
        self.game_state.apply_damage_to_enemy(7)
        self.assertEqual(self.game_state.enemy_damage, 7)
        
        # Apply more damage
        self.game_state.apply_damage_to_enemy(5)
        self.assertEqual(self.game_state.enemy_damage, 12)
        
    def test_is_enemy_defeated(self):
        """Test checking if the enemy is defeated"""
        # Enemy not defeated
        self.game_state.apply_damage_to_enemy(5)
        self.assertFalse(self.game_state.is_enemy_defeated())
        
        # Enemy defeated
        self.game_state.apply_damage_to_enemy(5)  # Total: 10
        self.assertTrue(self.game_state.is_enemy_defeated())
        
    def test_can_yield(self):
        """Test checking if a player can yield"""
        # Initially all players can yield
        self.assertTrue(self.game_state.can_yield(self.player1))
        self.assertTrue(self.game_state.can_yield(self.player2))
        
        # Player2 yields
        self.game_state.yields_in_a_row["Player2"] = True
        
        # Player1 should still be able to yield
        self.assertTrue(self.game_state.can_yield(self.player1))
        
        # Player1 also yields
        self.game_state.yields_in_a_row["Player1"] = True
        
        # Now neither player should be able to yield
        self.assertFalse(self.game_state.can_yield(self.player1))
        self.assertFalse(self.game_state.can_yield(self.player2))
        
    def test_record_yield(self):
        """Test recording a player's yield"""
        self.assertFalse(self.game_state.yields_in_a_row["Player1"])
        
        # Record yield
        self.game_state.record_yield(self.player1)
        self.assertTrue(self.game_state.yields_in_a_row["Player1"])
        
    def test_record_action(self):
        """Test recording a player's action (not yield)"""
        # Set player as yielded
        self.game_state.yields_in_a_row["Player1"] = True
        
        # Record action
        self.game_state.record_action(self.player1)
        self.assertFalse(self.game_state.yields_in_a_row["Player1"])
        
    def test_check_loss_all_players_dead(self):
        """Test checking for loss when all players are dead"""
        # Initially no loss
        self.assertFalse(self.game_state.check_loss())
        
        # Kill all players
        self.player1.alive = False
        self.player2.alive = False
        
        # Should be a loss
        self.assertTrue(self.game_state.check_loss())
        
    def test_check_loss_no_enemy(self):
        """Test checking for loss when there's no enemy (game won)"""
        # Set no current enemy (game won)
        self.game_state.current_enemy = None
        
        # Should not be a loss
        self.assertFalse(self.game_state.check_loss())
        
    def test_is_game_won(self):
        """Test checking if the game is won"""
        # Initially not won
        self.assertFalse(self.game_state.is_game_won())
        
        # Set no current enemy (game won)
        self.game_state.current_enemy = None
        
        # Should be won
        self.assertTrue(self.game_state.is_game_won())
        
    def test_repr(self):
        """Test the __repr__ method"""
        # Get string representation
        repr_str = repr(self.game_state)
        
        # Check that it contains key information
        self.assertIn("Jack of Hearts", repr_str)
        self.assertIn("Damage: 0", repr_str)
        self.assertIn("Shield: 0", repr_str)
        self.assertIn("Player1", repr_str)
        self.assertIn("Player2", repr_str)
        self.assertIn("Tavern size: 2", repr_str)
        self.assertIn("Discard: 0", repr_str)

if __name__ == '__main__':
    unittest.main()