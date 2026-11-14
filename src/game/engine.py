from typing import List

from cli.display import (
    show_current_state, show_player_hand, show_invalid_action, show_win, show_loss,
    show_discard_phase, show_played_cards, show_jester_played, show_enemy_defeated,
    show_new_enemy, show_warning, show_damage_dealt
)
from cli.input_handler import (
    prompt_card_indexes_to_play, prompt_card_indexes_to_discard, prompt_next_player
)
from game.card import Card
from game.deck import TavernDeck, CastleDeck, DiscardPile
from game.enemy import Enemy
from game.game_state import GameState
from game.player import Player
from game.rules import total_attack, is_enemy_immune, is_valid_play, exact_defeat, resolve_suit_power, can_play_jester


def create_enemy_faces():
    suits = ('hearts', 'diamonds', 'clubs', 'spades')
    face_stats = {
        'J': {'attack': 10, 'health': 20, 'name': 'Jack'},
        'Q': {'attack': 15, 'health': 30, 'name': 'Queen'},
        'K': {'attack': 20, 'health': 40, 'name': 'King'},
    }
    jacks, queens, kings = [], [], []
    for rank, coll in zip(('J', 'Q', 'K'), (jacks, queens, kings)):
        stat = face_stats[rank]
        for suit in suits:
            coll.append(
                Enemy(
                    suit=suit,
                    rank=rank,
                    value=stat['attack'],
                    c_type='enemy',
                    display_name=f"{stat['name']} of {suit.capitalize()}",
                    health=stat['health'],
                    attack=stat['attack']
                )
            )
    return jacks, queens, kings


def deal_initial_hands(game_state: GameState):
    players = list(game_state.players)
    for _ in range(players[0].hand_limit):
        for p in players:
            card = game_state.tavern_deck.draw_one()
            if card:
                p.add_card(card)


class Engine:
    def __init__(self, logger, get_logger):
        self.last_action_result = None
        self.phase = None
        self.game_state = None
        self.done = False
        self.logger = logger
        self.get_logger = get_logger

    def log_data_state(self, game_state: GameState):
        for p in game_state.players:
            self.logger.data(f"HAND {p.name}: {[str(c) for c in p.hand]}")
        self.logger.data(f"TAVERN DECK: {[str(c) for c in game_state.tavern_deck.cards]}")
        self.logger.data(f"CASTLE DECK: {[str(c) for c in game_state.castle_deck.cards]}")
        self.logger.data(f"DISCARD PILE: {[str(c) for c in game_state.discard_pile.cards]}")
        self.logger.data(f"UPCOMING ENEMIES: {[str(c) for c in game_state.castle_deck.cards]}")

    def handle_jester_played(self, active_player: Player, game_state: GameState):
        show_jester_played()
        self.logger.info(f"Jester played by {active_player.name}")
        game_state.face_jester_played = True
        game_state.update_enemy_immunity()
        # Official rules: spades played against a spades enemy BEFORE the
        # Jester (blocked by immunity) begin reducing the enemy's attack once
        # the Jester negates immunity — apply them retroactively, then clear.
        if game_state.current_enemy.suit == 'spades' and game_state.pending_spades:
            for spade_card in game_state.pending_spades:
                resolve_suit_power('spades', spade_card.value, game_state.current_enemy, discard_pile=None,
                                   tavern_deck=None, players=None, game_state=game_state,
                                   acting_player=active_player)
            game_state.pending_spades.clear()
        next_player = prompt_next_player(list(game_state.players))
        self.logger.info(f"Next player after Jester: {next_player.name}")
        while game_state.get_active_player() != next_player:
            game_state.advance_turn()

    def defense_validation(self, game_state: GameState, needed: int):
        active_player = game_state.get_active_player()
        if active_player.hand_size() == 0:
            show_loss(f"{active_player.name} has no cards left to discard and is killed by the enemy!")
            obs = self.get_observation()
            return False, obs, {"result": "player_defeated"}
        if sum(c.value for c in active_player.hand) < needed:
            show_loss(
                f"{active_player.name} cannot discard enough cards to cover the enemy's attack of {needed}!")
            obs = self.get_observation()
            return False, obs, {"result": "player_defeated"}
        return True, None, {}

    def handle_enemy_defense(self, game_state: GameState, discards: List[Card]):
        game_state.get_active_player().play_cards(discards)
        for d in discards:
            game_state.discard_pile.add_to_top(d)
        return 'continue'

    def handle_enemy_defeat(self, game_state: GameState):
        if exact_defeat(game_state.current_enemy.current_damage, game_state.current_enemy):
            show_enemy_defeated(game_state.current_enemy, exact=True)
            self.logger.info(f"Enemy {game_state.current_enemy.display_name} defeated EXACTLY.")
            game_state.tavern_deck.add_to_top(game_state.current_enemy)
        else:
            show_enemy_defeated(game_state.current_enemy, exact=False)
            self.logger.info(f"Enemy {game_state.current_enemy.display_name} defeated.")
            game_state.discard_pile.add_to_top(game_state.current_enemy)
        for c in game_state.cards_against_enemy:
            game_state.discard_pile.add_to_top(c)
        game_state.cards_against_enemy.clear()
        game_state.pending_spades.clear()
        next_enemy = game_state.castle_deck.draw_one()
        self.logger.info(f"Next enemy: {next_enemy.display_name if next_enemy else 'None (game won)'}")
        game_state.current_enemy = next_enemy
        game_state.face_jester_played = False
        # Reset immunity for the new battle too: after a jester-assisted
        # defeat the stale enemy_immunity=False would keep the pending-spades
        # collection gate from ever firing against the next spades enemy.
        game_state.enemy_immunity = True
        if next_enemy is not None:
            show_new_enemy()

    def handle_suit_powers(self, cards_played: List[Card], game_state: GameState, active_player: Player
                           , total_value: int):
        for suit in ('hearts', 'diamonds', 'spades'):
            if any(c.suit == suit for c in cards_played):
                any_card_of_suit = next(c for c in cards_played if c.suit == suit)
                if not is_enemy_immune(game_state.current_enemy, any_card_of_suit, game_state.face_jester_played):
                    self.logger.info(f"Resolving suit power: {suit} for {active_player.name} (value {total_value})")
                    resolve_suit_power(
                        suit, total_value, game_state.current_enemy, game_state.discard_pile,
                        game_state.tavern_deck, list(game_state.players), game_state, active_player
                    )

    def handle_yield_action(self, active_player: Player, game_state: GameState):
        self.logger.info(f"{active_player.name} yields.")
        self.logger.data(f"YIELD by {active_player.name}")
        if not game_state.can_yield(active_player):
            show_warning(f"{active_player.name} cannot yield at this time!")
            self.logger.error(f"{active_player.name} attempted to yield but cannot yield now.")
            return ''
        game_state.record_yield(active_player)
        self.log_data_state(game_state)
        return 'continue'

    # ------------------------------------------------------------------
    # Core turn engine, shared by the CLI (run) and gym (step) front-ends.
    # Core methods return outcome kinds; rewards, obs tuples, prompts,
    # logging and show_* display calls stay in the front-end wrappers.
    # ------------------------------------------------------------------

    def _begin_turn(self, game_state: GameState):
        """Turn-start bookkeeping: record the turn start, then check whether
        the active player is stuck (empty hand AND cannot yield — everyone
        loses). Returns 'player_defeated' or None."""
        active_player = game_state.get_active_player()
        game_state.record_turn_start(active_player)
        # Stuck check placed after record_turn_start: record_turn_start
        # resets only THIS player's own flag, while can_yield (inside
        # player_is_stuck) reads only the OTHER players' flags — so this
        # ordering cannot change whether the player is stuck.
        if game_state.player_is_stuck(active_player):
            return 'player_defeated'
        return None

    def _attempt_yield(self, active_player: Player, game_state: GameState):
        """Shared yield: can_yield gate, record the yield, then check whether
        the enemy attack forces a defense phase or the turn advances (phase
        and turn mutations included). Returns 'forbidden',
        'defense_required' or 'turn_advanced'."""
        if not game_state.can_yield(active_player):
            return 'forbidden'
        game_state.record_yield(active_player)
        if game_state.current_enemy.effective_attack() > 0:
            self.phase = "defense"
            return 'defense_required'
        game_state.advance_turn()
        self.phase = "attack"
        return 'turn_advanced'

    def _execute_play(self, active_player: Player, cards: List[Card], game_state: GameState):
        """Shared play execution: membership validation, spade-pending
        collection (enemy-suit spades while immunity blocks them), the
        attack, and the defeat/win checks. Does NOT advance the turn or set
        the phase — front-ends do that from the outcome. Returns
        'invalid_play', 'jester', 'enemy_defeated', 'win',
        'defense_required' or 'turn_advanced'."""
        if not all(c in active_player.hand for c in cards):
            return 'invalid_play'
        if not is_valid_play(cards):
            return 'invalid_play'
        if can_play_jester(cards):
            self.handle_jester_played(active_player, game_state)
            return 'jester'
        for c in cards:
            if c.suit == 'spades' and game_state.current_enemy.suit == 'spades' and game_state.enemy_immunity:
                game_state.pending_spades.append(c)
        self.handle_player_attack(active_player, cards, game_state)
        if game_state.current_enemy.is_defeated():
            self.handle_enemy_defeat(game_state)
            # Official rules Step 3 (iv): the player who defeated the enemy skips Step 4 and begins a new turn against the enemy just revealed — turn does NOT advance here.
            if game_state.is_game_won():
                return 'win'
            return 'enemy_defeated'
        if game_state.current_enemy.effective_attack() > 0:
            return 'defense_required'
        return 'turn_advanced'

    def _execute_defense(self, active_player: Player, discards: List[Card], game_state: GameState):
        """Shared defense execution: defense_validation, discard membership
        and coverage checks, then the discard. Returns ('loss', obs, info),
        ('invalid_defense', None, None) or ('success', None, None). Turn
        advancement stays with the front-ends."""
        needed = game_state.current_enemy.effective_attack()
        defense_validation = self.defense_validation(game_state, needed)
        if not defense_validation[0]:
            return 'loss', defense_validation[1], defense_validation[2]
        if not all(c in active_player.hand for c in discards):
            return 'invalid_defense', None, None
        if sum(c.value for c in discards) < needed:
            return 'invalid_defense', None, None
        self.handle_enemy_defense(game_state, discards)
        return 'success', None, None

    def handle_play_action(self, active_player: Player, game_state: GameState):
        while True:
            cards_played = prompt_card_indexes_to_play(active_player)
            # If no cards are played, treat as yield
            if not cards_played:
                if self.handle_yield_action(active_player, game_state):
                    return 'yielded'
            if is_valid_play(cards_played):
                self.logger.info(f"{active_player.name} plays: {[str(c) for c in cards_played]}")
                self.logger.data(f"CARD_PLAYED by {active_player.name}: {[str(c) for c in cards_played]}")
                break
            else:
                show_invalid_action("That is not a valid play. Try again.")
                self.logger.warning(f"Invalid play by {active_player.name}: {[str(c) for c in cards_played]}")
        if can_play_jester(cards_played):
            self.handle_jester_played(active_player, game_state)
            return 'jester'
        for c in cards_played:
            if c.suit == game_state.current_enemy.suit and game_state.enemy_immunity:
                show_warning(f"You have played a card of the enemy's suit ({c.suit}) while immunity is ON")
        outcome = self._execute_play(active_player, cards_played, game_state)
        if outcome == 'jester':
            return 'jester'
        if outcome == 'win':
            return 'game_won'
        if outcome == 'enemy_defeated':
            return 'enemy_defeated'
        # 'defense_required'/'turn_advanced': run()'s loop performs the
        # effective-attack check and turn handling itself. 'invalid_play' is
        # unreachable here (the prompt loop already guaranteed a valid
        # in-hand play) — treat it like any non-terminal result.
        return 'continue'

    def handle_player_attack(self, active_player: Player, cards_played: List[Card], game_state: GameState):
        active_player.play_cards(cards_played)
        game_state.cards_against_enemy.extend(cards_played)
        show_played_cards(cards_played)
        total_value = sum(c.value for c in cards_played)
        self.handle_suit_powers(cards_played, game_state, active_player, total_value)
        damage = total_attack(cards_played, game_state.face_jester_played, enemy=game_state.current_enemy)
        game_state.current_enemy.apply_damage(damage)
        self.logger.info(
            f"{active_player.name} dealt {damage} damage to enemy {game_state.current_enemy.display_name} (total damage: {game_state.current_enemy.current_damage})")
        show_damage_dealt(damage)

    def run(self, players: List[Player]):
        self.logger.info("Game started.")
        num_players = len(players)
        jacks, queens, kings = create_enemy_faces()
        self.logger.info("Castle deck created and shuffled.")
        castle_deck = CastleDeck(jacks, queens, kings)
        tavern_deck = TavernDeck(num_players)
        self.logger.info("Tavern deck created and shuffled.")
        discard_pile = DiscardPile()
        self.game_state = GameState(players=players, tavern_deck=tavern_deck, castle_deck=castle_deck,
                                    discard_pile=discard_pile)
        self.logger.info("Game state initialized.")
        deal_initial_hands(self.game_state)
        self.logger.info("Initial hands dealt.")
        self.phase = 'attack'
        while not self.game_state.is_game_won():
            active_player = self.game_state.get_active_player()
            stuck = self._begin_turn(self.game_state) if self.phase == 'attack' else None
            self.logger.info(f"Turn: {active_player.name}")
            self.log_data_state(self.game_state)
            if stuck:
                show_loss(f"{active_player.name} cannot play a card or yield — game lost!")
                self.logger.error(f"{active_player.name} is stuck (no cards to play, cannot yield): game lost.")
                return
            if active_player.hand_size() == 0:
                # Empty hand but yield allowed (not stuck): auto-yield — never
                # prompt with an empty hand. With attack > 0 the empty hand
                # then dies in defense_validation (per rules); with attack 0
                # the turn simply advances.
                self.logger.info(f"{active_player.name} has no cards left and yields automatically.")
                self.logger.data(f"YIELD by {active_player.name}")
                outcome = self._attempt_yield(active_player, self.game_state)
                if outcome == 'turn_advanced':
                    self.log_data_state(self.game_state)
                    continue
                # 'defense_required': fall through — the empty hand dies in
                # defense_validation (per rules).
            show_current_state(self.game_state)
            show_player_hand(active_player)
            while True:
                if self.phase == 'attack':
                    result = self.handle_play_action(active_player, self.game_state)
                    if result in ('jester', 'enemy_defeated'):
                        break
                    if result == 'game_won':
                        # FIX-7: the win is terminal — break cleanly and let
                        # the outer while's not is_game_won() condition exit
                        # so show_win() runs (previously this fell through to
                        # current_enemy.effective_attack() with current_enemy
                        # None and crashed with AttributeError).
                        break
                    enemy_attack = self.game_state.current_enemy.effective_attack()
                    if enemy_attack > 0:
                        self.phase = 'defense'
                        continue
                    else:
                        self.game_state.advance_turn()
                        self.log_data_state(self.game_state)
                        self.phase = 'attack'
                        break
                elif self.phase == 'defense':
                    # Defense phase: handle enemy attack
                    needed = self.game_state.current_enemy.effective_attack()
                    show_discard_phase(self.game_state.get_active_player(), needed)
                    if active_player.hand_size() == 0:
                        # Empty hand (e.g. after an Option-A defeat with the
                        # new enemy attacking): never prompt — every index
                        # would be invalid and empty input would re-prompt
                        # forever. defense_validation in the core handles the
                        # loss.
                        outcome, loss_obs, loss_info = self._execute_defense(
                            active_player, [], self.game_state)
                    else:
                        outcome, loss_obs, loss_info = self._execute_defense(
                            active_player,
                            prompt_card_indexes_to_discard(active_player, needed),
                            self.game_state)
                    if outcome == 'loss':
                        return loss_obs, -10, True, loss_info
                    self.game_state.advance_turn()
                    self.log_data_state(self.game_state)
                    self.phase = 'attack'
                    break
            if self.game_state.check_loss():
                show_loss("A player could not satisfy enemy attack or no legal plays remain.")
                self.logger.error("Game lost: a player could not satisfy enemy attack or no legal plays remain.")
                return
        show_win()
        self.logger.info("Game won!")

    def reset(self, players: List[Player]):
        # Initialize game state and phase for Gym interface
        num_players = len(players)
        jacks, queens, kings = create_enemy_faces()
        castle_deck = CastleDeck(jacks, queens, kings)
        tavern_deck = TavernDeck(num_players)
        discard_pile = DiscardPile()
        self.game_state = GameState(players=players, tavern_deck=tavern_deck, castle_deck=castle_deck,
                                    discard_pile=discard_pile)
        deal_initial_hands(self.game_state)
        self.phase = "attack"
        self.last_action_result = None
        self.done = False
        return self.get_observation()

    def step(self, action):
        # Gym-style step: process one phase per call
        if self.done:
            raise Exception("Game is over. Call reset().")
        obs, reward, done, info = None, 0, False, {}
        if self.phase == "attack":
            stuck = self._begin_turn(self.game_state)
            if stuck:
                # Preempts action parsing: a stuck player has no legal action,
                # so terminate instead of looping -1 rewards forever. The
                # epilogue below applies the standard -0.01 dispatch penalty.
                obs = self.get_observation()
                reward, done, info = -10, True, {"result": "player_defeated"}
            else:
                obs, reward, done, info = self.step_attack(action)
        elif self.phase == "defense":
            obs, reward, done, info = self.step_defense(action)
        else:
            raise Exception(f"Unknown phase: {self.phase}")
        self.done = done
        step_penalty = 0.01
        return obs, reward - step_penalty, done, info

    def step_attack(self, action):
        game_state = self.game_state
        active_player = game_state.get_active_player()
        if action["type"] == "play":
            cards_played = action["cards"]
            # If no cards are played, treat as yield
            if not cards_played:
                outcome = self._attempt_yield(active_player, game_state)
                if outcome == 'forbidden':
                    obs = self.get_observation()
                    return obs, -1, False, {"error": "Yield forbidden"}
                self.logger.info(f"{active_player.name} yields (no cards played).")
                self.logger.data(f"YIELD by {active_player.name}")
                if outcome == 'defense_required':
                    obs = self.get_observation()
                    return obs, -0.1, False, {"result": "yielded_defense_required"}
                obs = self.get_observation()
                return obs, -0.1, False, {"result": "yielded_turn_advanced"}
            outcome = self._execute_play(active_player, cards_played, game_state)
            if outcome == 'invalid_play':
                obs = self.get_observation()
                return obs, -1, False, {"error": "Invalid play"}
            if outcome == 'jester':
                obs = self.get_observation()
                return obs, 0.2, False, {"result": "jester"}
            if outcome == 'win':
                obs = self.get_observation()
                return obs, 10, True, {"result": "win"}
            if outcome == 'enemy_defeated':
                obs = self.get_observation()
                return obs, 2, False, {"result": "enemy_defeated"}
            if outcome == 'defense_required':
                self.phase = "defense"
                obs = self.get_observation()
                return obs, 0.2, False, {"result": "defense_required"}
            # 'turn_advanced'
            game_state.advance_turn()
            self.phase = "attack"
            obs = self.get_observation()
            return obs, 0.2, False, {"result": "turn_advanced"}
        obs = self.get_observation()
        return obs, -1, False, {"error": "Unknown action type"}

    def step_defense(self, action):
        game_state = self.game_state
        active_player = game_state.get_active_player()
        if action["type"] == "defend":
            outcome, loss_obs, loss_info = self._execute_defense(active_player, action["cards"], game_state)
            if outcome == 'loss':
                return loss_obs, -10, True, loss_info
            if outcome == 'invalid_defense':
                obs = self.get_observation()
                return obs, -1, False, {"error": "Invalid defense"}
            print(f"{active_player.name} discarded {len(action['cards'])} card(s) to cover attack.")
            # After defense, always advance turn
            game_state.advance_turn()
            self.phase = "attack"
            obs = self.get_observation()
            return obs, 0.2, False, {"result": "defense_success"}
        obs = self.get_observation()
        return obs, -1, False, {"error": "Unknown defense action"}

    def get_observation(self):
        # Return a dict with game state and current phase
        return {
            "phase": self.phase,
            "game_state": self.game_state,  # You may want to serialize this for RL
        }
