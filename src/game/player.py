class Player:
    def __init__(self, name, hand_limit):
        self.name = name
        self.hand = []  # list of Card objects
        self.hand_limit = hand_limit
        self.alive = True
        self.has_yielded = False  # for yield-chain rule

    def add_card(self, card):
        if len(self.hand) < self.hand_limit:
            self.hand.append(card)
            return True
        return False

    def play_cards(self, cards):
        # Removes a list of cards from hand; validates presence
        for card in cards:
            if card not in self.hand:
                return False  # Invalid play: not all cards in hand
        for card in cards:
            self.hand.remove(card)
        return True

    def show_hand(self):
        return [c.display_name for c in self.hand]

    def hand_size(self):
        return len(self.hand)

    def is_alive(self):
        return self.alive

    def __repr__(self):
        return f"Player({self.name}, hand={self.show_hand()})"
