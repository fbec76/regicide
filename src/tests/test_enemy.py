import unittest
from src.game.enemy import Enemy

class TestEnemy(unittest.TestCase):
    def setUp(self):
        # Create sample enemies for testing
        self.jack = Enemy("hearts", "J", 10, "enemy", "Jack of Hearts", 20, 10)
        self.queen = Enemy("diamonds", "Q", 15, "enemy", "Queen of Diamonds", 30, 15)
        self.king = Enemy("clubs", "K", 20, "enemy", "King of Clubs", 40, 20)
        
    def test_enemy_initialization(self):
        """Test that enemies are initialized with correct attributes"""
        # Test Jack
        self.assertEqual(self.jack.suit, "hearts")
        self.assertEqual(self.jack.rank, "J")
        self.assertEqual(self.jack.value, 10)
        self.assertEqual(self.jack.c_type, "enemy")
        self.assertEqual(self.jack.display_name, "Jack of Hearts")
        self.assertEqual(self.jack.max_health, 20)
        self.assertEqual(self.jack.current_damage, 0)
        self.assertEqual(self.jack.shield, 0)
        self.assertEqual(self.jack.attack, 10)
        self.assertFalse(self.jack.suit_immunity_cancelled)
        
        # Test Queen
        self.assertEqual(self.queen.max_health, 30)
        self.assertEqual(self.queen.attack, 15)
        
        # Test King
        self.assertEqual(self.king.max_health, 40)
        self.assertEqual(self.king.attack, 20)
        
    def test_apply_damage(self):
        """Test applying damage to an enemy"""
        self.jack.apply_damage(5)
        self.assertEqual(self.jack.current_damage, 5)
        
        # Apply more damage
        self.jack.apply_damage(10)
        self.assertEqual(self.jack.current_damage, 15)
        
    def test_apply_shield(self):
        """Test applying shield to an enemy"""
        self.jack.apply_shield(3)
        self.assertEqual(self.jack.shield, 3)
        
        # Apply more shield
        self.jack.apply_shield(2)
        self.assertEqual(self.jack.shield, 5)
        
    def test_reset_shield(self):
        """Test resetting an enemy's shield"""
        self.jack.apply_shield(5)
        self.assertEqual(self.jack.shield, 5)
        
        self.jack.reset_shield()
        self.assertEqual(self.jack.shield, 0)
        
    def test_is_defeated(self):
        """Test checking if an enemy is defeated"""
        # Enemy not defeated
        self.jack.apply_damage(10)
        self.assertFalse(self.jack.is_defeated())
        
        # Enemy defeated
        self.jack.apply_damage(10)
        self.assertTrue(self.jack.is_defeated())
        
        # Enemy exactly at max health
        self.queen.apply_damage(30)
        self.assertTrue(self.queen.is_defeated())
        
    def test_effective_attack(self):
        """Test calculating an enemy's effective attack with shield"""
        # No shield
        self.assertEqual(self.jack.effective_attack(), 10)
        
        # With shield
        self.jack.apply_shield(3)
        self.assertEqual(self.jack.effective_attack(), 7)
        
        # Shield greater than attack
        self.jack.apply_shield(10)
        self.assertEqual(self.jack.effective_attack(), 0)
        
    def test_reset(self):
        """Test resetting an enemy to initial state"""
        # Apply damage, shield, and cancel immunity
        self.jack.apply_damage(10)
        self.jack.apply_shield(5)
        self.jack.cancel_immunity()
        
        # Verify changes
        self.assertEqual(self.jack.current_damage, 10)
        self.assertEqual(self.jack.shield, 5)
        self.assertTrue(self.jack.suit_immunity_cancelled)
        
        # Reset
        self.jack.reset()
        
        # Verify reset
        self.assertEqual(self.jack.current_damage, 0)
        self.assertEqual(self.jack.shield, 0)
        self.assertFalse(self.jack.suit_immunity_cancelled)
        
    def test_cancel_immunity(self):
        """Test cancelling an enemy's suit immunity"""
        # Initially immunity is active
        self.assertFalse(self.jack.suit_immunity_cancelled)
        self.assertTrue(self.jack.has_immunity())
        
        # Cancel immunity
        self.jack.cancel_immunity()
        self.assertTrue(self.jack.suit_immunity_cancelled)
        self.assertFalse(self.jack.has_immunity())
        
    def test_has_immunity(self):
        """Test checking if an enemy has immunity"""
        # Initially immunity is active
        self.assertTrue(self.jack.has_immunity())
        
        # Cancel immunity
        self.jack.cancel_immunity()
        self.assertFalse(self.jack.has_immunity())
        
    def test_repr(self):
        """Test the __repr__ method"""
        # Test with default values
        expected = "Jack of Hearts: Health 20/20, Attack 10 (Base 10), Suit: hearts, Immunity: ON"
        self.assertEqual(repr(self.jack), expected)
        
        # Test with damage and shield
        self.jack.apply_damage(5)
        self.jack.apply_shield(3)
        expected = "Jack of Hearts: Health 15/20, Attack 7 (Base 10), Suit: hearts, Immunity: ON"
        self.assertEqual(repr(self.jack), expected)
        
        # Test with cancelled immunity
        self.jack.cancel_immunity()
        expected = "Jack of Hearts: Health 15/20, Attack 7 (Base 10), Suit: hearts, Immunity: OFF"
        self.assertEqual(repr(self.jack), expected)

if __name__ == '__main__':
    unittest.main()