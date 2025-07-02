class Card:
    def __init__(self, suit, rank, value, c_type, display_name):
        self.suit = suit
        self.rank = rank
        self.value = value
        self.c_type = c_type
        self.display_name = display_name

    def is_animal(self):
        return self.c_type == 'animal'

    def is_jester(self):
        return self.c_type == 'jester'

    def is_enemy(self):
        return self.c_type == 'enemy'

    def is_face(self, card):
        return self.rank == card.rank and self.suit == card.suit

    def __repr__(self):
        return f"{self.display_name} of {self.suit}"