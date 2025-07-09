from game.card import Card

class Enemy(Card):
    def __init__(self, suit, rank, value, c_type, display_name, health, attack):
        super().__init__(suit, rank, value, c_type, display_name)
        self.max_health = health   # 20/30/40 for J/Q/K
        self.current_damage = 0
        self.shield = 0           # total shields (from Spades)
        self.attack = attack      # 10/15/20 for J/Q/K

    def apply_damage(self, amount):
        self.current_damage += amount

    def apply_shield(self, amount):
        self.shield += amount

    def is_defeated(self):
        return self.current_damage >= self.max_health

    def effective_attack(self):
        return max(self.attack - self.shield, 0)

    def __repr__(self):
        return (f"{self.display_name}: "
                f"Health {self.max_health - self.current_damage}/{self.max_health}, "
                f"Attack {self.effective_attack()} (Base {self.attack}), "
                f"Suit: {self.suit}, ")