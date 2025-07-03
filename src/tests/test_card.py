import unittest
from src.game.card import Card

class TestCard(unittest.TestCase):
    def setUp(self):
        # Create sample cards for testing
        self.number_card = Card("hearts", "7", 7, "number", "7 of Hearts")
        self.animal_card = Card("spades", "A", 1, "animal", "Animal of Spades")
        self.jester_card = Card("joker", "Jester", 0, "jester", "Jester")
        self.enemy_card = Card("clubs", "K", 10, "enemy", "King of Clubs")
        
    def test_card_initialization(self):
        """Test that cards are initialized with correct attributes"""
        self.assertEqual(self.number_card.suit, "hearts")
        self.assertEqual(self.number_card.rank, "7")
        self.assertEqual(self.number_card.value, 7)
        self.assertEqual(self.number_card.c_type, "number")
        self.assertEqual(self.number_card.display_name, "7 of Hearts")
        
    def test_is_animal(self):
        """Test the is_animal method"""
        self.assertFalse(self.number_card.is_animal())
        self.assertTrue(self.animal_card.is_animal())
        self.assertFalse(self.jester_card.is_animal())
        self.assertFalse(self.enemy_card.is_animal())
        
    def test_is_jester(self):
        """Test the is_jester method"""
        self.assertFalse(self.number_card.is_jester())
        self.assertFalse(self.animal_card.is_jester())
        self.assertTrue(self.jester_card.is_jester())
        self.assertFalse(self.enemy_card.is_jester())
        
    def test_is_enemy(self):
        """Test the is_enemy method"""
        self.assertFalse(self.number_card.is_enemy())
        self.assertFalse(self.animal_card.is_enemy())
        self.assertFalse(self.jester_card.is_enemy())
        self.assertTrue(self.enemy_card.is_enemy())
        
    def test_is_face(self):
        """Test the is_face method"""
        # Create a duplicate card
        duplicate_card = Card("hearts", "7", 7, "number", "7 of Hearts")
        # Create a different card with same rank but different suit
        different_suit = Card("diamonds", "7", 7, "number", "7 of Diamonds")
        # Create a different card with same suit but different rank
        different_rank = Card("hearts", "8", 8, "number", "8 of Hearts")
        
        self.assertTrue(self.number_card.is_face(duplicate_card))
        self.assertFalse(self.number_card.is_face(different_suit))
        self.assertFalse(self.number_card.is_face(different_rank))
        
    def test_repr(self):
        """Test the __repr__ method"""
        self.assertEqual(repr(self.number_card), "7 of Hearts")
        self.assertEqual(repr(self.animal_card), "Animal of Spades")
        self.assertEqual(repr(self.jester_card), "Jester")
        self.assertEqual(repr(self.enemy_card), "King of Clubs")

if __name__ == '__main__':
    unittest.main()