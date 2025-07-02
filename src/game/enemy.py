from .card import Card

class Enemy(Card):
    def __init__(self, suit, rank, value, c_type, display_name, health, attack):
        super().__init__(suit, rank, value, c_type, display_name)
        self.max_health = health   # 20/30/40 for J/Q/K
        self.current_damage = 0
        self.shield = 0           # total shields (from Spades)
        self.attack = attack      # 10/15/20 for J/Q/K
        self.suit_immunity_cancelled = False

    def apply_damage(self, amount):
        self.current_damage += amount

    def apply_shield(self, amount):
        self.shield += amount

    def reset_shield(self):
        self.shield = 0

    def is_defeated(self):
        return self.current_damage >= self.max_health

    def effective_attack(self):
        return max(self.attack - self.shield, 0)

    def reset(self):
        self.current_damage = 0
        self.shield = 0
        self.suit_immunity_cancelled = False

    def cancel_immunity(self):
        self.suit_immunity_cancelled = True

    def has_immunity(self):
        return not self.suit_immunity_cancelled

    def __repr__(self):
        return (f"{self.display_name}: "
                f"Health {self.max_health - self.current_damage}/{self.max_health}, "
                f"Attack {self.effective_attack()} (Base {self.attack}), "
                f"Suit: {self.suit}, "
                f"Immunity: {'OFF' if self.suit_immunity_cancelled else 'ON'}")