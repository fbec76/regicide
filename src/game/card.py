class Card:
    def __init__(self, suit, rank, value, c_type, display_name):
        self.suit = suit
        self.rank = rank
        self.value = value
        self.c_type = c_type
        self.display_name = display_name

    def __repr__(self):
        return f"{self.display_name} of {self.suit}"