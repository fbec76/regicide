import unittest
from src.game.rules import (
    valid_combo, can_play_jester, valid_animal_pair, is_valid_play,
    is_enemy_immune, resolve_suit_power, total_attack, yield_allowed,
    must_handle_enemy_attack, exact_defeat
)
from src.game.card import Card
from src.game.enemy import Enemy
from src.game.deck import Deck, DiscardPile
from src.game.player import Player
from src.game.game_state import GameState
from unittest.mock import MagicMock, patch

class TestRules(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.number_card1 = Card("hearts", "5", 5, "number", "5 of Hearts")
        self.number_card2 = Card("diamonds", "5", 5, "number", "5 of Diamonds")
        self.number_card3 = Card("clubs", "5", 5, "number", "5 of Clubs")
        self.number_card4 = Card("spades", "5", 5, "number", "5 of Spades")
        
        self.animal_card = Card("hearts", "A", 1, "animal", "Animal of Hearts")
        self.jester_card = Card("joker", "Jester", 0, "jester", "Jester")
        
        # Create enemies for testing
        self.hearts_enemy = Enemy("hearts", "J", 10, "enemy", "Jack of Hearts", 20, 10)
        self.clubs_enemy = Enemy("clubs", "Q", 15, "enemy", "Queen of Clubs", 30, 15)
        
    def test_valid_combo(self):
        """Test valid combo detection"""
        # Valid 2-card combo
        combo = [self.number_card1, self.number_card2]
        self.assertTrue(valid_combo(combo))
        
        # Valid 3-card combo
        combo = [self.number_card1, self.number_card2, self.number_card3]
        self.assertTrue(valid_combo(combo))
        
        # Valid 4-card combo
        combo = [self.number_card1, self.number_card2, self.number_card3, self.number_card4]
        self.assertTrue(valid_combo(combo))
        
        # Invalid: too few cards
        combo = [self.number_card1]
        self.assertFalse(valid_combo(combo))
        
        # Invalid: too many cards
        combo = [self.number_card1, self.number_card2, self.number_card3, self.number_card4, self.number_card1]
        self.assertFalse(valid_combo(combo))
        
        # Invalid: contains animal
        combo = [self.number_card1, self.animal_card]
        self.assertFalse(valid_combo(combo))
        
        # Invalid: contains jester
        combo = [self.number_card1, self.jester_card]
        self.assertFalse(valid_combo(combo))
        
        # Invalid: different values
        different_value = Card("hearts", "6", 6, "number", "6 of Hearts")
        combo = [self.number_card1, different_value]
        self.assertFalse(valid_combo(combo))
        
        # Invalid: sum > 10
        high_value1 = Card("hearts", "6", 6, "number", "6 of Hearts")
        high_value2 = Card("diamonds", "6", 6, "number", "6 of Diamonds")
        combo = [high_value1, high_value2]  # Sum = 12
        self.assertFalse(valid_combo(combo))
        
    def test_can_play_jester(self):
        """Test jester play validation"""
        # Valid: single jester
        self.assertTrue(can_play_jester([self.jester_card]))
        
        # Invalid: not a jester
        self.assertFalse(can_play_jester([self.number_card1]))
        
        # Invalid: multiple cards
        self.assertFalse(can_play_jester([self.jester_card, self.jester_card]))
        
    def test_valid_animal_pair(self):
        """Test animal pair validation"""
        # Valid: two animals
        two_animals = [self.animal_card, Card("diamonds", "A", 1, "animal", "Animal of Diamonds")]
        self.assertTrue(valid_animal_pair(two_animals))
        
        # Valid: animal + number
        animal_number = [self.animal_card, self.number_card1]
        self.assertTrue(valid_animal_pair(animal_number))
        
        # Invalid: animal + jester
        animal_jester = [self.animal_card, self.jester_card]
        self.assertFalse(valid_animal_pair(animal_jester))
        
        # Invalid: two numbers
        two_numbers = [self.number_card1, self.number_card2]
        self.assertFalse(valid_animal_pair(two_numbers))
        
        # Invalid: not 2 cards
        self.assertFalse(valid_animal_pair([self.animal_card]))
        self.assertFalse(valid_animal_pair([self.animal_card, self.number_card1, self.number_card2]))
        
    def test_is_valid_play(self):
        """Test overall play validation"""
        # Valid: single card
        self.assertTrue(is_valid_play([self.number_card1]))
        self.assertTrue(is_valid_play([self.animal_card]))
        self.assertTrue(is_valid_play([self.jester_card]))
        
        # Valid: combo
        combo = [self.number_card1, self.number_card2]
        self.assertTrue(is_valid_play(combo))
        
        # Valid: animal pair
        animal_pair = [self.animal_card, self.number_card1]
        self.assertTrue(is_valid_play(animal_pair))
        
        # Invalid: not a valid play type
        invalid_play = [self.number_card1, self.animal_card, self.jester_card]
        self.assertFalse(is_valid_play(invalid_play))
        
    def test_is_enemy_immune(self):
        """Test enemy immunity checking"""
        # Enemy is immune to its own suit
        self.assertTrue(is_enemy_immune(self.hearts_enemy, self.number_card1))  # hearts enemy vs hearts card
        self.assertFalse(is_enemy_immune(self.hearts_enemy, self.number_card2))  # hearts enemy vs diamonds card
        
        # Jester negates immunity
        self.assertFalse(is_enemy_immune(self.hearts_enemy, self.number_card1, jester_active=True))
        
    def test_resolve_suit_power(self):
        """Test suit power resolution"""
        # Setup for testing
        discard_pile = DiscardPile([Card("clubs", "2", 2, "number", "2 of Clubs")])
        tavern_deck = Deck([Card("diamonds", "3", 3, "number", "3 of Diamonds")])
        player1 = Player("Player1", 7)
        player2 = Player("Player2", 7)
        players = [player1, player2]
        game_state = MagicMock()
        
        # Test Hearts power (healing)
        with patch.object(discard_pile, 'shuffle_and_return_n_cards') as mock_shuffle:
            mock_shuffle.return_value = [Card("clubs", "2", 2, "number", "2 of Clubs")]
            resolve_suit_power(self.number_card1, self.hearts_enemy, discard_pile, tavern_deck, players, game_state, player1)
            mock_shuffle.assert_called_once_with(5)  # Should try to heal 5 cards (card value)
            
        # Test Diamonds power (drawing)
        self.assertEqual(len(player1.hand), 0)
        self.assertEqual(len(player2.hand), 0)
        resolve_suit_power(self.number_card2, self.hearts_enemy, discard_pile, tavern_deck, players, game_state, player1)
        self.assertEqual(len(player1.hand) + len(player2.hand), 1)  # Should have drawn 1 card (only 1 in tavern)
        
        # Test Spades power (shielding)
        resolve_suit_power(self.number_card4, self.hearts_enemy, discard_pile, tavern_deck, players, game_state, player1)
        self.assertEqual(self.hearts_enemy.shield, 5)  # Should add shield equal to card value
        
    def test_total_attack(self):
        """Test attack calculation"""
        # Basic attack
        cards = [self.number_card1, self.number_card2]  # 5 + 5 = 10
        self.assertEqual(total_attack(cards), 10)
        
        # With clubs (doubled)
        cards = [self.number_card1, self.number_card3]  # 5 + (5*2) = 15
        self.assertEqual(total_attack(cards), 15)
        
        # Against clubs enemy (no doubling unless jester active)
        cards = [self.number_card1, self.number_card3]  # 5 + 5 = 10
        self.assertEqual(total_attack(cards, jester_active=False, enemy=self.clubs_enemy), 10)
        
        # Against clubs enemy with jester active (doubling applies)
        cards = [self.number_card1, self.number_card3]  # 5 + (5*2) = 15
        self.assertEqual(total_attack(cards, jester_active=True, enemy=self.clubs_enemy), 15)
        
        # Jester doesn't contribute to attack
        cards = [self.number_card1, self.jester_card]  # 5 + 0 = 5
        self.assertEqual(total_attack(cards), 5)
        
    def test_yield_allowed(self):
        """Test yield permission checking"""
        game_state = MagicMock()
        player = MagicMock()
        
        # Yield allowed
        game_state.can_yield.return_value = True
        self.assertTrue(yield_allowed(game_state, player))
        
        # Yield not allowed
        game_state.can_yield.return_value = False
        self.assertFalse(yield_allowed(game_state, player))
        
    def test_must_handle_enemy_attack(self):
        """Test enemy attack calculation"""
        # No shield
        self.assertEqual(must_handle_enemy_attack(self.hearts_enemy, MagicMock()), 10)
        
        # With shield
        self.hearts_enemy.apply_shield(3)
        self.assertEqual(must_handle_enemy_attack(self.hearts_enemy, MagicMock()), 7)
        
    def test_exact_defeat(self):
        """Test exact defeat checking"""
        # Not exact
        self.assertFalse(exact_defeat(15, self.hearts_enemy))  # 15 vs 20 health
        
        # Exact
        self.assertTrue(exact_defeat(20, self.hearts_enemy))  # 20 vs 20 health
        
        # Over
        self.assertFalse(exact_defeat(25, self.hearts_enemy))  # 25 vs 20 health

if __name__ == '__main__':
    unittest.main()