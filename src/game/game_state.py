from collections import deque

class GameState:
    def __init__(self, players, tavern_deck, castle_deck, discard_pile):
        """
        players: list of Player objects (in turn order)
        tavern_deck: TavernDeck object
        castle_deck: CastleDeck object
        discard_pile: DiscardPile (or Deck)
        """
        self.players = deque(players)  # Turn order rotation: leftmost is active turn
        self.tavern_deck = tavern_deck
        self.castle_deck = castle_deck
        self.discard_pile = discard_pile

        self.current_enemy = self.castle_deck.draw_one()  # Card object
        self.enemy_damage = 0
        self.spade_shield = 0
        self.enemy_immunity = True   # Until Jester played (if active, disables enemy's suit power immunity)
        self.face_jester_played = False  # Tracks temporary immunity negation

        # For yielding rules
        self.yields_in_a_row = {p.name: False for p in self.players}
        self.round_history = []

        # For combo play
        self.played_this_enemy = []  # Cards played by all players against this enemy

    def start_turn(self):
        """Prepare state for a new turn."""
        self.spade_shield = 0
        self.enemy_damage = 0
        self.face_jester_played = False
        self.played_this_enemy.clear()
        self.update_enemy_immunity()

    def get_active_player(self):
        return self.players[0]

    def advance_turn(self):
        self.players.rotate(-1)  # Move to the next player

    def defeat_enemy(self, exact=False):
        """
        Called if enemy is defeated.
        - exact: If True and damage == health, enemy is shuffled to bottom of tavern, not discard pile.
        """
        defeated_enemy = self.current_enemy
        if exact:
            # Place facedown on top of tavern deck (not visible) per rule
            self.tavern_deck.add_to_top(defeated_enemy)
        else:
            self.discard_pile.add_to_top(defeated_enemy)
        # Discard all played cards from pile
        for card in self.played_this_enemy:
            self.discard_pile.add_to_top(card)
        self.played_this_enemy.clear()
        # Get next enemy, if available
        self.current_enemy = self.castle_deck.draw_one()
        self.enemy_damage = 0
        self.spade_shield = 0
        self.face_jester_played = False
        self.update_enemy_immunity()

    def update_enemy_immunity(self):
        # By default, immunity is on unless a Jester has been played against this enemy
        if self.face_jester_played:
            self.enemy_immunity = False
        else:
            self.enemy_immunity = True

    def play_card_against_enemy(self, player, cards_played, suit_powers_effect):
        """
        This function should:
        - Accept the player and their cards played (single card, combo, or animal pair)
        - Add to the play pile and remove from hand
        - Apply card numbers to enemy damage (clubbing for double, etc)
        - Record suit power effects, shields, and that cards were played
        """
        # Assume cards_played is a list of Card
        player.play_cards(cards_played)
        self.played_this_enemy.extend(cards_played)
        # Suit power handling is external
        # You may want to trigger effects here or in main engine
        # e.g., self.apply_suit_powers(cards_played) if desired

    def apply_spades_shield(self, amount):
        self.spade_shield += amount

    def apply_damage_to_enemy(self, amount):
        self.enemy_damage += amount

    def is_enemy_defeated(self):
        return self.enemy_damage >= self.current_enemy.value  # Enemy card .value holds attack or health by your Card design

    def can_yield(self, player):
        # At least one other player must not have yielded on their last turn
        return not all([v for k, v in self.yields_in_a_row.items() if k != player.name])

    def record_yield(self, player):
        self.yields_in_a_row[player.name] = True

    def record_action(self, player):
        self.yields_in_a_row[player.name] = False

    def check_loss(self):
        # Loss conditions: player unable to discard for damage; no legal play/yield
        alive = any(p.is_alive() for p in self.players)
        if not alive:
            return True
        if self.current_enemy is None:
            # Game already won (handled externally)
            return False
        # TO DO: Add more rule checks as your engine is built
        return False

    def is_game_won(self):
        return self.current_enemy is None

    def __repr__(self):
        state = f"Enemy: {self.current_enemy}, Damage: {self.enemy_damage}, Shield: {self.spade_shield}\n"
        for p in self.players:
            state += f"{p.name}: {p.show_hand()} (Hand: {len(p.hand)})\n"
        state += f"Tavern size: {len(self.tavern_deck)}, Discard: {len(self.discard_pile)}"
        return state