import unittest
from unittest.mock import patch
from src.game.player import Player
from src.game.card import Card

class TestPlayer(unittest.TestCase):
    def setUp(self):
        # Create a player for testing
        self.player = Player("TestPlayer", 7)
        
        # Create sample cards for testing
        self.card1 = Card("hearts", "7", 7, "number", "7 of Hearts")
        self.card2 = Card("diamonds", "8", 8, "number", "8 of Diamonds")
        self.card3 = Card("clubs", "9", 9, "number", "9 of Clubs")
        self.card4 = Card("spades", "10", 10, "number", "10 of Spades")
        
    def test_player_initialization(self):
        """Test that player is initialized with correct attributes"""
        self.assertEqual(self.player.name, "TestPlayer")
        self.assertEqual(self.player.hand, [])
        self.assertEqual(self.player.hand_limit, 7)
        self.assertTrue(self.player.alive)
        self.assertFalse(self.player.has_yielded)
        
    def test_draw_cards(self):
        """Test drawing cards up to hand limit"""
        # Draw 3 cards
        self.player.draw_cards([self.card1, self.card2, self.card3])
        self.assertEqual(len(self.player.hand), 3)
        self.assertEqual(self.player.hand[0], self.card1)
        self.assertEqual(self.player.hand[1], self.card2)
        self.assertEqual(self.player.hand[2], self.card3)
        
        # Draw more cards up to hand limit
        cards_to_draw = [Card("hearts", str(i), i, "number", f"{i} of Hearts") for i in range(2, 7)]
        self.player.draw_cards(cards_to_draw)
        self.assertEqual(len(self.player.hand), 7)  # Should be at hand limit
        
        # Try to draw more cards beyond hand limit
        self.player.draw_cards([self.card4])
        self.assertEqual(len(self.player.hand), 7)  # Should still be at hand limit
        
    def test_add_card(self):
        """Test adding a single card to hand"""
        # Add a card to empty hand
        result = self.player.add_card(self.card1)
        self.assertTrue(result)
        self.assertEqual(len(self.player.hand), 1)
        self.assertEqual(self.player.hand[0], self.card1)
        
        # Fill hand to limit
        for i in range(2, 8):
            card = Card("hearts", str(i), i, "number", f"{i} of Hearts")
            result = self.player.add_card(card)
            self.assertTrue(result)
        
        # Try to add card beyond limit
        result = self.player.add_card(self.card4)
        self.assertFalse(result)
        self.assertEqual(len(self.player.hand), 7)  # Should still be at hand limit
        
    def test_remove_card(self):
        """Test removing a card from hand"""
        # Add cards to hand
        self.player.draw_cards([self.card1, self.card2, self.card3])
        
        # Remove a card
        result = self.player.remove_card(self.card2)
        self.assertTrue(result)
        self.assertEqual(len(self.player.hand), 2)
        self.assertNotIn(self.card2, self.player.hand)
        
        # Try to remove a card not in hand
        result = self.player.remove_card(self.card4)
        self.assertFalse(result)
        self.assertEqual(len(self.player.hand), 2)
        
    def test_play_cards(self):
        """Test playing multiple cards from hand"""
        # Add cards to hand
        self.player.draw_cards([self.card1, self.card2, self.card3, self.card4])
        
        # Play two cards
        result = self.player.play_cards([self.card1, self.card3])
        self.assertTrue(result)
        self.assertEqual(len(self.player.hand), 2)
        self.assertNotIn(self.card1, self.player.hand)
        self.assertNotIn(self.card3, self.player.hand)
        self.assertIn(self.card2, self.player.hand)
        self.assertIn(self.card4, self.player.hand)
        
        # Try to play a card not in hand
        new_card = Card("hearts", "A", 1, "animal", "Animal of Hearts")
        result = self.player.play_cards([self.card2, new_card])
        self.assertFalse(result)
        self.assertEqual(len(self.player.hand), 2)  # Hand should be unchanged
        
    @patch('builtins.input', side_effect=['0', '1'])
    def test_discard_for_damage_success(self, mock_input):
        """Test discarding cards for damage (successful case)"""
        # Add cards to hand
        self.player.draw_cards([self.card1, self.card2])  # Total value: 15
        
        # Discard for damage less than total value
        success, discarded = self.player.discard_for_damage(10)
        
        # Check result
        self.assertTrue(success)
        self.assertEqual(len(discarded), 2)
        self.assertIn(self.card1, discarded)
        self.assertIn(self.card2, discarded)
        self.assertEqual(len(self.player.hand), 0)
        
    @patch('builtins.input', side_effect=['0'])
    def test_discard_for_damage_failure(self, mock_input):
        """Test discarding cards for damage (failure case)"""
        # Add card to hand
        self.player.draw_cards([self.card1])  # Value: 7
        
        # Try to discard for damage more than total value
        success, discarded = self.player.discard_for_damage(10)
        
        # Check result
        self.assertFalse(success)
        self.assertEqual(len(discarded), 1)
        self.assertIn(self.card1, discarded)
        self.assertEqual(len(self.player.hand), 0)
        self.assertFalse(self.player.alive)  # Player should be marked as not alive
        
    def test_show_hand(self):
        """Test getting a list of card display names in hand"""
        # Add cards to hand
        self.player.draw_cards([self.card1, self.card2])
        
        # Check hand display
        hand_display = self.player.show_hand()
        self.assertEqual(len(hand_display), 2)
        self.assertEqual(hand_display[0], "7 of Hearts")
        self.assertEqual(hand_display[1], "8 of Diamonds")
        
    def test_hand_size(self):
        """Test getting the number of cards in hand"""
        self.assertEqual(self.player.hand_size(), 0)
        
        # Add cards to hand
        self.player.draw_cards([self.card1, self.card2])
        self.assertEqual(self.player.hand_size(), 2)
        
    def test_is_alive(self):
        """Test checking if player is alive"""
        self.assertTrue(self.player.is_alive())
        
        # Mark player as not alive
        self.player.alive = False
        self.assertFalse(self.player.is_alive())
        
    def test_reset_yield(self):
        """Test resetting player's yield status"""
        # Set player as yielded
        self.player.has_yielded = True
        self.assertTrue(self.player.has_yielded)
        
        # Reset yield
        self.player.reset_yield()
        self.assertFalse(self.player.has_yielded)
        
    def test_repr(self):
        """Test the __repr__ method"""
        # Empty hand
        expected = "Player(TestPlayer, hand=[])"
        self.assertEqual(repr(self.player), expected)
        
        # With cards in hand
        self.player.draw_cards([self.card1, self.card2])
        expected = "Player(TestPlayer, hand=['7 of Hearts', '8 of Diamonds'])"
        self.assertEqual(repr(self.player), expected)

if __name__ == '__main__':
    unittest.main()