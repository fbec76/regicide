from cli.display import (
    show_current_state, show_player_hand, show_invalid_action, show_win, show_loss,
    show_yield_forbidden, show_discard_phase, show_played_cards, show_jester_played
)
from cli.input_handler import (
    prompt_action_choice, prompt_card_indexes_to_play, prompt_card_indexes_to_discard, prompt_next_player
)
from game.deck import TavernDeck, CastleDeck, DiscardPile
from game.enemy import Enemy
from game.game_state import GameState
from game.rules import total_attack, is_enemy_immune, is_valid_play, exact_defeat, resolve_suit_power, can_play_jester


class Engine:
    def __init__(self, logger, get_logger):
        self.logger = logger
        self.get_logger = get_logger

    def create_enemy_faces(self):
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

    def deal_initial_hands(self, game_state):
        players = list(game_state.players)
        for _ in range(players[0].hand_limit):
            for p in players:
                card = game_state.tavern_deck.draw_one()
                if card:
                    p.add_card(card)

    def handle_enemy_attack(self, game_state, player):
        enemy_attack = game_state.current_enemy.effective_attack()
        if enemy_attack > 0:
            show_discard_phase(player, enemy_attack)
            if player.hand_size() == 0:
                show_loss(f"{player.name} has no cards left to discard and is killed by the enemy!")
                return False
            if sum(c.value for c in player.hand) < enemy_attack:
                show_loss(f"{player.name} cannot discard enough cards to cover the enemy's attack of {enemy_attack}!")
                return False
            discards = prompt_card_indexes_to_discard(player, enemy_attack)
            player.play_cards(discards)
            for d in discards:
                game_state.discard_pile.add_to_top(d)
            print(f"{player.name} discarded {len(discards)} card(s) to cover attack.")
            return True
        return True

    def log_data_state(self, game_state):
        for p in game_state.players:
            self.logger.data(f"HAND {p.name}: {[str(c) for c in p.hand]}")
        self.logger.data(f"TAVERN DECK: {[str(c) for c in game_state.tavern_deck.cards]}")
        self.logger.data(f"CASTLE DECK: {[str(c) for c in game_state.castle_deck.cards]}")
        self.logger.data(f"DISCARD PILE: {[str(c) for c in game_state.discard_pile.cards]}")
        self.logger.data(f"UPCOMING ENEMIES: {[str(c) for c in game_state.castle_deck.cards]}")

    def _handle_play_action(self, active_player, game_state, played_spades):
        while True:
            cards_played = prompt_card_indexes_to_play(active_player)
            if is_valid_play(cards_played):
                self.logger.info(f"{active_player.name} plays: {[str(c) for c in cards_played]}")
                self.logger.data(f"CARD_PLAYED by {active_player.name}: {[str(c) for c in cards_played]}")
                break
            else:
                show_invalid_action("That is not a valid play. Try again.")
                self.logger.warning(f"Invalid play by {active_player.name}: {[str(c) for c in cards_played]}")
        if can_play_jester(cards_played):
            show_jester_played()
            self.logger.info(f"Jester played by {active_player.name}")
            game_state.face_jester_played = True
            game_state.update_enemy_immunity()
            if game_state.current_enemy.suit == 'spades' and played_spades:
                for spade_card in played_spades:
                    resolve_suit_power('spades', spade_card.value, game_state.current_enemy, discard_pile=None,
                                       tavern_deck=None, players=None, game_state=game_state,
                                       acting_player=active_player)
                played_spades.clear()
            next_player = prompt_next_player(list(game_state.players))
            self.logger.info(f"Next player after Jester: {next_player.name}")
            while game_state.get_active_player() != next_player:
                game_state.advance_turn()
            return 'jester'
        for c in cards_played:
            if c.suit == game_state.current_enemy.suit and game_state.enemy_immunity:
                confirm = input(
                    f"You are playing a card of the enemy's suit ({c.suit}) while immunity is ON. Sure? (y/N): ").strip().lower()
                if confirm != 'y':
                    show_invalid_action("Play cancelled by player due to immunity confirmation.")
                    self.logger.info(f"{active_player.name} cancelled play due to immunity warning.")
                    cards_played.clear()
                    break
            if c.suit == 'spades' and game_state.current_enemy.suit == 'spades' and game_state.enemy_immunity:
                played_spades.append(c)
        if not cards_played:
            return 'cancelled'
        active_player.play_cards(cards_played)
        game_state.played_this_enemy.extend(cards_played)
        show_played_cards(cards_played)
        total_value = sum(c.value for c in cards_played)
        for suit in ('hearts', 'diamonds', 'spades'):
            if any(c.suit == suit for c in cards_played):
                any_card_of_suit = next(c for c in cards_played if c.suit == suit)
                if not is_enemy_immune(game_state.current_enemy, any_card_of_suit, game_state.face_jester_played):
                    self.logger.info(f"Resolving suit power: {suit} for {active_player.name} (value {total_value})")
                    resolve_suit_power(
                        suit, total_value, game_state.current_enemy, game_state.discard_pile,
                        game_state.tavern_deck, list(game_state.players), game_state, active_player
                    )
        damage = total_attack(cards_played, game_state.face_jester_played, enemy=game_state.current_enemy)
        game_state.current_enemy.apply_damage(damage)
        self.logger.info(
            f"{active_player.name} dealt {damage} damage to enemy {game_state.current_enemy.display_name} (total damage: {game_state.current_enemy.current_damage})")
        print(
            f"\033[93mDamage dealt to enemy: {damage}. (Total: {game_state.current_enemy.current_damage})\033[0m")
        if game_state.current_enemy.is_defeated():
            if exact_defeat(game_state.current_enemy.current_damage, game_state.current_enemy):
                print("You have defeated the enemy EXACTLY! Place enemy on top of Tavern deck.")
                self.logger.info(f"Enemy {game_state.current_enemy.display_name} defeated EXACTLY.")
                game_state.tavern_deck.add_to_top(game_state.current_enemy)
            else:
                print("Enemy defeated! Card goes to discard pile.")
                self.logger.info(f"Enemy {game_state.current_enemy.display_name} defeated.")
                game_state.discard_pile.add_to_top(game_state.current_enemy)
            for c in game_state.played_this_enemy:
                game_state.discard_pile.add_to_top(c)
            game_state.played_this_enemy.clear()
            next_enemy = game_state.castle_deck.draw_one()
            self.logger.info(f"Next enemy: {next_enemy.display_name if next_enemy else 'None (game won)'}")
            game_state.current_enemy = next_enemy
            game_state.face_jester_played = False
            if next_enemy is not None:
                print("\nA new enemy appears!\n")
            return 'enemy_defeated'
        else:
            if not self.handle_enemy_attack(game_state, active_player):
                self.logger.warning(f"{active_player.name} was defeated by enemy attack.")
                return 'player_defeated'
        return 'continue'

    def _handle_yield_action(self, active_player, game_state):
        self.logger.info(f"{active_player.name} yields.")
        self.logger.data(f"YIELD by {active_player.name}")
        if not self.handle_enemy_attack(game_state, active_player):
            self.logger.warning(f"{active_player.name} was defeated by enemy attack after yielding.")
            return 'player_defeated'
        game_state.advance_turn()
        self.log_data_state(game_state)
        return 'continue'

    def run(self, players):
        self.logger.info("Game started.")
        num_players = len(players)
        jacks, queens, kings = self.create_enemy_faces()
        self.logger.info("Castle deck created and shuffled.")
        castle_deck = CastleDeck(jacks, queens, kings)
        tavern_deck = TavernDeck(num_players)
        self.logger.info("Tavern deck created and shuffled.")
        discard_pile = DiscardPile()
        game_state = GameState(players=players, tavern_deck=tavern_deck, castle_deck=castle_deck,
                               discard_pile=discard_pile)
        self.logger.info("Game state initialized.")
        self.deal_initial_hands(game_state)
        self.logger.info("Initial hands dealt.")
        while not game_state.is_game_won():
            active_player = game_state.get_active_player()
            self.logger.info(f"Turn: {active_player.name}")
            self.log_data_state(game_state)
            show_current_state(game_state)
            show_player_hand(active_player)
            if active_player.hand_size() == 0:
                show_loss(f"{active_player.name} has no cards left to play and is defeated!")
                self.logger.info(f"{active_player.name} has no cards left to play and is defeated!")
                return
            if game_state.can_yield(active_player):
                action = prompt_action_choice(active_player)
            else:
                show_yield_forbidden()
                action = 'play'
            played_spades = []
            if action == 'play':
                result = self._handle_play_action(active_player, game_state, played_spades)
                if result in ('jester', 'cancelled', 'enemy_defeated'):
                    continue
                elif result == 'player_defeated':
                    return
                game_state.advance_turn()
                self.log_data_state(game_state)
            elif action == 'yield':
                result = self._handle_yield_action(active_player, game_state)
                if result == 'player_defeated':
                    return
            else:
                show_invalid_action("Unknown action. Skipping turn for safety.")
                self.logger.warning(f"Unknown action for {active_player.name}: {action}")
                game_state.advance_turn()
            if game_state.check_loss():
                show_loss("A player could not satisfy enemy attack or no legal plays remain.")
                self.logger.error("Game lost: a player could not satisfy enemy attack or no legal plays remain.")
                return
        show_win()
        self.logger.info("Game won!")
