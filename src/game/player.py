class Player:
    def __init__(self, name, hand_limit):
        self.name = name
        self.hand = []  # list of Card objects
        self.hand_limit = hand_limit
        self.alive = True
        self.has_yielded = False  # for yield-chain rule

    def draw_cards(self, cards):
        # Accepts a list of Card objects; enforces max hand size
        for card in cards:
            if len(self.hand) < self.hand_limit:
                self.hand.append(card)
            else:
                break  # Do not overdraw

    def add_card(self, card):
        if len(self.hand) < self.hand_limit:
            self.hand.append(card)
            return True
        return False

    def remove_card(self, card):
        # Removes a specific card object from hand (by identity)
        if card in self.hand:
            self.hand.remove(card)
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

    def discard_for_damage(self, value_needed):
        """
        Prompts the player to choose cards to discard for covering damage.
        Returns tuple: (success, list_of_discarded_cards)
        (This is a placeholder: final implementation could be UI/auto/AI)
        """
        print(f"{self.name}, you must discard cards totaling at least {value_needed}.")
        print("Your hand:", [f"{i}:{c.display_name}" for i,c in enumerate(self.hand)])
        running_sum = 0
        discards = []
        used_indexes = set()
        while running_sum < value_needed and len(used_indexes) < len(self.hand):
            try:
                idx = int(input(f"Choose card index to discard (running sum: {running_sum}): "))
                if idx in used_indexes or idx < 0 or idx >= len(self.hand):
                    print("Invalid or already chosen; pick again.")
                    continue
                card = self.hand[idx]
                running_sum += card.value
                discards.append(card)
                used_indexes.add(idx)
            except Exception:
                print("Enter a valid integer index!")
        # Finalize discard
        for card in discards:
            self.hand.remove(card)
        if running_sum >= value_needed:
            return True, discards
        else:
            self.alive = False
            return False, discards

    def show_hand(self):
        return [c.display_name for c in self.hand]

    def hand_size(self):
        return len(self.hand)

    def is_alive(self):
        return self.alive

    def reset_yield(self):
        self.has_yielded = False

    def __repr__(self):
        return f"Player({self.name}, hand={self.show_hand()})"