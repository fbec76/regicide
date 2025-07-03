import unittest
from unittest.mock import patch
from src.game.deck import Deck, CastleDeck, TavernDeck, DiscardPile
from src.game.card import Card

class TestDeck(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.card1 = Card("hearts", "7", 7, "number", "7 of Hearts")
        self.card2 = Card("diamonds", "8", 8, "number", "8 of Diamonds")
        self.card3 = Card("clubs", "9", 9, "number", "9 of Clubs")
        
        # Create a deck with the sample cards
        self.deck = Deck([self.card1, self.card2, self.card3])
        
    def test_deck_initialization(self):
        """Test that deck is initialized with correct cards"""
        self.assertEqual(len(self.deck.cards), 3)
        self.assertEqual(self.deck.cards[0], self.card1)
        self.assertEqual(self.deck.cards[1], self.card2)
        self.assertEqual(self.deck.cards[2], self.card3)
        
        # Test empty deck initialization
        empty_deck = Deck()
        self.assertEqual(len(empty_deck.cards), 0)
        
    def test_shuffle(self):
        """Test that shuffle randomizes the order of cards"""
        # Since shuffle is random, we'll mock random.shuffle to ensure predictable behavior
        with patch('random.shuffle') as mock_shuffle:
            self.deck.shuffle()
            mock_shuffle.assert_called_once_with(self.deck.cards)
            
    def test_draw(self):
        """Test drawing multiple cards from the deck"""
        drawn = self.deck.draw(2)
        self.assertEqual(len(drawn), 2)
        self.assertEqual(drawn[0], self.card1)
        self.assertEqual(drawn[1], self.card2)
        self.assertEqual(len(self.deck.cards), 1)
        self.assertEqual(self.deck.cards[0], self.card3)
        
    def test_draw_one(self):
        """Test drawing a single card from the deck"""
        card = self.deck.draw_one()
        self.assertEqual(card, self.card1)
        self.assertEqual(len(self.deck.cards), 2)
        
        # Test drawing from empty deck
        empty_deck = Deck()
        self.assertIsNone(empty_deck.draw_one())
        
    def test_add_to_top(self):
        """Test adding a card to the top of the deck"""
        new_card = Card("spades", "10", 10, "number", "10 of Spades")
        self.deck.add_to_top(new_card)
        self.assertEqual(len(self.deck.cards), 4)
        self.assertEqual(self.deck.cards[0], new_card)
        
    def test_add_to_bottom(self):
        """Test adding a card to the bottom of the deck"""
        new_card = Card("spades", "10", 10, "number", "10 of Spades")
        self.deck.add_to_bottom(new_card)
        self.assertEqual(len(self.deck.cards), 4)
        self.assertEqual(self.deck.cards[3], new_card)
        
    def test_is_empty(self):
        """Test checking if the deck is empty"""
        self.assertFalse(self.deck.is_empty())
        
        empty_deck = Deck()
        self.assertTrue(empty_deck.is_empty())
        
    def test_size(self):
        """Test getting the size of the deck"""
        self.assertEqual(self.deck.size(), 3)
        
        empty_deck = Deck()
        self.assertEqual(empty_deck.size(), 0)
        
    def test_peek_top(self):
        """Test peeking at the top card without removing it"""
        top_card = self.deck.peek_top()
        self.assertEqual(top_card, self.card1)
        self.assertEqual(len(self.deck.cards), 3)  # Card should still be in deck
        
        # Test peeking empty deck
        empty_deck = Deck()
        self.assertIsNone(empty_deck.peek_top())
        
    def test_len(self):
        """Test the __len__ method"""
        self.assertEqual(len(self.deck), 3)
        
        empty_deck = Deck()
        self.assertEqual(len(empty_deck), 0)


class TestCastleDeck(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.jack1 = Card("hearts", "J", 10, "enemy", "Jack of Hearts")
        self.jack2 = Card("diamonds", "J", 10, "enemy", "Jack of Diamonds")
        self.queen1 = Card("clubs", "Q", 15, "enemy", "Queen of Clubs")
        self.queen2 = Card("spades", "Q", 15, "enemy", "Queen of Spades")
        self.king1 = Card("hearts", "K", 20, "enemy", "King of Hearts")
        self.king2 = Card("diamonds", "K", 20, "enemy", "King of Diamonds")
        
    @patch('random.shuffle')
    def test_castle_deck_creation(self, mock_shuffle):
        """Test that castle deck is created correctly"""
        jacks = [self.jack1, self.jack2]
        queens = [self.queen1, self.queen2]
        kings = [self.king1, self.king2]
        
        castle_deck = CastleDeck(jacks, queens, kings)
        
        # Check that shuffle was called for each rank
        self.assertEqual(mock_shuffle.call_count, 3)
        
        # Check that the deck has all cards in the correct order (jacks, queens, kings)
        self.assertEqual(len(castle_deck.cards), 6)
        # Since we mocked shuffle, the order should be as provided
        self.assertIn(self.jack1, castle_deck.cards[:2])
        self.assertIn(self.jack2, castle_deck.cards[:2])
        self.assertIn(self.queen1, castle_deck.cards[2:4])
        self.assertIn(self.queen2, castle_deck.cards[2:4])
        self.assertIn(self.king1, castle_deck.cards[4:])
        self.assertIn(self.king2, castle_deck.cards[4:])


class TestTavernDeck(unittest.TestCase):
    @patch('random.shuffle')
    def test_tavern_deck_creation_2_players(self, mock_shuffle):
        """Test that tavern deck is created correctly for 2 players"""
        tavern_deck = TavernDeck(2)
        
        # Check that shuffle was called
        mock_shuffle.assert_called_once()
        
        # For 2 players, should have:
        # - 36 number cards (9 values × 4 suits)
        # - 4 animal cards (1 per suit)
        # - 0 jesters
        self.assertEqual(len(tavern_deck.cards), 40)
        
        # Count card types
        number_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'number')
        animal_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'animal')
        jester_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'jester')
        
        self.assertEqual(number_cards, 36)
        self.assertEqual(animal_cards, 4)
        self.assertEqual(jester_cards, 0)
        
    @patch('random.shuffle')
    def test_tavern_deck_creation_4_players(self, mock_shuffle):
        """Test that tavern deck is created correctly for 4 players"""
        tavern_deck = TavernDeck(4)
        
        # Check that shuffle was called
        mock_shuffle.assert_called_once()
        
        # For 4 players, should have:
        # - 36 number cards (9 values × 4 suits)
        # - 4 animal cards (1 per suit)
        # - 2 jesters
        self.assertEqual(len(tavern_deck.cards), 42)
        
        # Count card types
        number_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'number')
        animal_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'animal')
        jester_cards = sum(1 for card in tavern_deck.cards if card.c_type == 'jester')
        
        self.assertEqual(number_cards, 36)
        self.assertEqual(animal_cards, 4)
        self.assertEqual(jester_cards, 2)


class TestDiscardPile(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.card1 = Card("hearts", "7", 7, "number", "7 of Hearts")
        self.card2 = Card("diamonds", "8", 8, "number", "8 of Diamonds")
        self.card3 = Card("clubs", "9", 9, "number", "9 of Clubs")
        self.card4 = Card("spades", "10", 10, "number", "10 of Spades")
        
        # Create a discard pile with the sample cards
        self.discard_pile = DiscardPile([self.card1, self.card2, self.card3, self.card4])
        
    @patch('random.shuffle')
    def test_shuffle_and_return_n_cards(self, mock_shuffle):
        """Test shuffling and returning n cards from the discard pile"""
        # Return 2 cards
        returned_cards = self.discard_pile.shuffle_and_return_n_cards(2)
        
        # Check that shuffle was called
        mock_shuffle.assert_called_once()
        
        # Check that 2 cards were returned
        self.assertEqual(len(returned_cards), 2)
        
        # Check that the returned cards are no longer in the discard pile
        for card in returned_cards:
            self.assertNotIn(card, self.discard_pile.cards)
            
        # Check that the discard pile now has 2 cards
        self.assertEqual(len(self.discard_pile.cards), 2)
        
    @patch('random.shuffle')
    def test_shuffle_and_return_more_than_available(self, mock_shuffle):
        """Test returning more cards than available in the discard pile"""
        # Try to return 6 cards when only 4 are available
        returned_cards = self.discard_pile.shuffle_and_return_n_cards(6)
        
        # Check that shuffle was called
        mock_shuffle.assert_called_once()
        
        # Check that only 4 cards were returned
        self.assertEqual(len(returned_cards), 4)
        
        # Check that the discard pile is now empty
        self.assertEqual(len(self.discard_pile.cards), 0)


if __name__ == '__main__':
    unittest.main()