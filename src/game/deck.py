import random
from typing import List

from .card import Card

class Deck:
    def __init__(self, cards: List[Card]=None):
        self.cards = list(cards) if cards else []

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n=1):
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def draw_one(self):
        if self.cards:
            return self.cards.pop(0)
        return None

    def add_to_top(self, card):
        self.cards.insert(0, card)

    def add_to_bottom(self, card):
        self.cards.append(card)

    def is_empty(self):
        return len(self.cards) == 0

    def size(self):
        return len(self.cards)

    def peek_top(self):
        return self.cards[0] if self.cards else None

    def __len__(self):
        return len(self.cards)


class CastleDeck(Deck):
    """
    Supply four cards of each rank for Jacks/Queens/Kings, one per suit, as lists.
    """
    def __init__(self, jack_cards, queen_cards, king_cards):
        cards = CastleDeck.create_castle_stack(jack_cards, queen_cards, king_cards)
        super().__init__(cards)

    @staticmethod
    def create_castle_stack(jacks, queens, kings):
        random.shuffle(jacks)
        random.shuffle(queens)
        random.shuffle(kings)
        return jacks + queens + kings


class TavernDeck(Deck):
    """
    Compose of 2-10, Animals (Ace), and Jesters per # players.
    """
    def __init__(self, num_players):
        cards = TavernDeck.create_tavern_deck(num_players)
        super().__init__(cards)
        self.shuffle()

    @staticmethod
    def create_tavern_deck(num_players):
        suits = ('hearts', 'diamonds', 'clubs', 'spades')
        cards = []

        # 2-10 in all suits
        for value in range(2, 11):
            for suit in suits:
                display = f"{value} of {suit.capitalize()}"
                cards.append(Card(suit, str(value), value, 'number', display))

        # 4 Animal Companions (Aces, value 1)
        for suit in suits:
            display = f"Animal of {suit.capitalize()}"
            cards.append(Card(suit, 'A', 1, 'animal', display))

        # Jesters: 0 for 1-2; 1 for 3; 2 for 4 players
        num_jesters = {1: 0, 2: 0, 3: 1, 4: 2}[num_players]
        for _ in range(num_jesters):
            cards.append(Card('joker', 'Jester', 0, 'jester', 'Jester'))

        return cards

class DiscardPile(Deck):
    """
    Models the discard pile. Specialized for Heart power (healing).
    Remove N random cards to return to Tavern deck, as per Regicide rules.
    """
    def shuffle_and_return_n_cards(self, n):
        cards_copy = self.cards.copy()
        random.shuffle(cards_copy)
        # Return up to n (may be fewer if less in pile)
        taken = cards_copy[:n]
        for card in taken:
            self.cards.remove(card)
        return taken