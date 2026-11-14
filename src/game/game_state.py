from collections import deque

from game.player import Player


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

        self.total_turns = 0
        self.current_enemy = self.castle_deck.draw_one()  # Card object
        self.enemy_damage = 0
        self.spade_shield = 0
        self.enemy_immunity = True  # Until Jester played (if active, disables enemy's suit power immunity)
        self.face_jester_played = False  # Tracks temporary immunity negation

        # For yielding rules: maps player name -> whether that player yielded
        # on their most recent turn. record_turn_start resets a player's flag
        # to False when their turn begins; record_yield sets it to True when
        # they yield. can_yield consults the OTHER players' flags.
        self.yields_in_a_row = {p.name: False for p in self.players}

        # For combo play
        self.cards_against_enemy = []  # Cards played by all players against this enemy

        # Cards of the current enemy's suit (spades) whose suit power was
        # blocked by enemy immunity. The Jester retroactively applies them
        # ("spades played prior to the Jester will begin reducing the attack
        # value of the enemy"). Battle-scoped: cleared when the enemy is
        # defeated. Not related to the legacy write-only spade_shield field.
        self.pending_spades: list = []

    def get_active_player(self):
        return self.players[0]

    def advance_turn(self):
        self.players.rotate(-1)  # Move to the next player
        self.total_turns += 1


    def update_enemy_immunity(self):
        # By default, immunity is on unless a Jester has been played against this enemy
        if self.face_jester_played:
            self.enemy_immunity = False
        else:
            self.enemy_immunity = True

    def can_yield(self, player: Player):
        # At least one other player must not have yielded on their last turn
        return True if self.total_turns == 0 else not all([v for k, v in self.yields_in_a_row.items() if k != player.name])

    def record_turn_start(self, player: Player):
        # The player's turn (re)started without a yield yet this turn.
        self.yields_in_a_row[player.name] = False

    def record_yield(self, player: Player):
        # The player yielded on this turn.
        self.yields_in_a_row[player.name] = True

    def player_is_stuck(self, player: Player) -> bool:
        """Official rules: a player who can neither play a card nor yield on
        their turn loses the game for everyone. Stuck = empty hand AND yield
        forbidden. can_yield reads the OTHER players' flags, so
        record_turn_start on this player never changes stuck-ness."""
        return player.hand_size() == 0 and not self.can_yield(player)

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
