from dotenv import load_dotenv

load_dotenv()

from __init__ import get_logger

logger = get_logger("regicide.main")
from game.rules import total_attack, is_enemy_immune, is_valid_play, exact_defeat, resolve_suit_power, can_play_jester
from game.player import Player
from game.deck import TavernDeck, CastleDeck, DiscardPile
from game.enemy import Enemy
from game.game_state import GameState

# CLI
from cli.display import (
    show_current_state, show_player_hand, show_invalid_action, show_win, show_loss,
    show_yield_forbidden, show_discard_phase, show_played_cards, show_jester_played, show_welcome
)
from cli.input_handler import (
    prompt_action_choice, prompt_card_indexes_to_play, prompt_card_indexes_to_discard, prompt_next_player
)


#### SETUP HELPERS

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


def ask_player_names(num_players, max_hand_size):
    players = []
    for i in range(num_players):
        name = input(f"Name for player {i + 1}: ").strip() or f'Player{i + 1}'
        players.append(Player(name=name, hand_limit=max_hand_size))
    return players


def deal_initial_hands(game_state):
    # Rule: deal up to max hand size from the tavern, round-robin
    players = list(game_state.players)
    for _ in range(players[0].hand_limit):
        for p in players:
            card = game_state.tavern_deck.draw_one()
            if card:
                p.add_card(card)


def handle_enemy_attack(game_state, player):
    """Handle the enemy attacking the current player. Returns True if the player survived, False if not."""
    enemy_attack = game_state.current_enemy.effective_attack()
    if enemy_attack > 0:
        show_discard_phase(player, enemy_attack)
        if player.hand_size() == 0:
            show_loss(f"{player.name} has no cards left to discard and is killed by the enemy!")
            return False
        discards = prompt_card_indexes_to_discard(player, enemy_attack)
        player.play_cards(discards)
        for d in discards:
            game_state.discard_pile.add_to_top(d)
        print(f"{player.name} discarded {len(discards)} card(s) to cover attack.")
        return True
    return True


def log_data_state(game_state):
    # Log all players' hands
    for p in game_state.players:
        logger.data(f"HAND {p.name}: {[str(c) for c in p.hand]}")
    # Log all cards in decks (in order)
    logger.data(f"TAVERN DECK: {[str(c) for c in game_state.tavern_deck.cards]}")
    logger.data(f"CASTLE DECK: {[str(c) for c in game_state.castle_deck.cards]}")
    logger.data(f"DISCARD PILE: {[str(c) for c in game_state.discard_pile.cards]}")
    # Log upcoming enemies (all in castle deck)
    logger.data(f"UPCOMING ENEMIES: {[str(c) for c in game_state.castle_deck.cards]}")


#### MAIN GAME LOOP

def main():
    logger.info("Game started.")
    show_welcome()
    num_players = ask_player_count()

    hand_size_table = {2: 7, 3: 6, 4: 5}
    max_hand_size = hand_size_table[num_players]

    players = ask_player_names(num_players, max_hand_size)
    logger.info(f"Players: {[p.name for p in players]}")

    jacks, queens, kings = create_enemy_faces()
    logger.info("Castle deck created and shuffled.")
    castle_deck = CastleDeck(jacks, queens, kings)
    tavern_deck = TavernDeck(num_players)
    logger.info("Tavern deck created and shuffled.")
    discard_pile = DiscardPile()

    game_state = GameState(players=players, tavern_deck=tavern_deck, castle_deck=castle_deck, discard_pile=discard_pile)
    logger.info("Game state initialized.")

    deal_initial_hands(game_state)
    logger.info("Initial hands dealt.")

    while not game_state.is_game_won():
        active_player = game_state.get_active_player()
        logger.info(f"Turn: {active_player.name}")
        log_data_state(game_state)
        show_current_state(game_state)
        show_player_hand(active_player)

        if game_state.can_yield(active_player):
            action = prompt_action_choice(active_player)
        else:
            show_yield_forbidden()
            action = 'play'

        played_spades = []
        if action == 'play':
            while True:
                cards_played = prompt_card_indexes_to_play(active_player)
                if is_valid_play(cards_played):
                    logger.info(f"{active_player.name} plays: {[str(c) for c in cards_played]}")
                    logger.data(f"CARD_PLAYED by {active_player.name}: {[str(c) for c in cards_played]}")
                    break
                else:
                    show_invalid_action("That is not a valid play. Try again.")
                    logger.warning(f"Invalid play by {active_player.name}: {[str(c) for c in cards_played]}")

            if can_play_jester(cards_played):
                show_jester_played()
                logger.info(f"Jester played by {active_player.name}")
                game_state.face_jester_played = True
                game_state.update_enemy_immunity()
                if game_state.current_enemy.suit == 'spades' and played_spades:
                    for spade_card in played_spades:
                        resolve_suit_power('spades', spade_card.value, game_state.current_enemy, discard_pile=None,
                                           tavern_deck=None, players=None, game_state=game_state,
                                           acting_player=active_player)
                    played_spades.clear()
                next_player = prompt_next_player(list(game_state.players))
                logger.info(f"Next player after Jester: {next_player.name}")
                while game_state.get_active_player() != next_player:
                    game_state.advance_turn()
                continue

            for c in cards_played:
                if c.suit == game_state.current_enemy.suit and game_state.enemy_immunity:
                    confirm = input(
                        f"You are playing a card of the enemy's suit ({c.suit}) while immunity is ON. Sure? (y/N): ").strip().lower()
                    if confirm != 'y':
                        show_invalid_action("Play cancelled by player due to immunity confirmation.")
                        logger.info(f"{active_player.name} cancelled play due to immunity warning.")
                        cards_played.clear()
                        break
                if c.suit == 'spades' and game_state.current_enemy.suit == 'spades' and game_state.enemy_immunity:
                    played_spades.append(c)
            if not cards_played:
                continue

            active_player.play_cards(cards_played)
            game_state.played_this_enemy.extend(cards_played)
            show_played_cards(cards_played)

            total_value = sum(c.value for c in cards_played)
            for suit in ('hearts', 'diamonds', 'spades'):
                if any(c.suit == suit for c in cards_played):
                    any_card_of_suit = next(c for c in cards_played if c.suit == suit)
                    if not is_enemy_immune(game_state.current_enemy, any_card_of_suit, game_state.face_jester_played):
                        logger.info(f"Resolving suit power: {suit} for {active_player.name} (value {total_value})")
                        resolve_suit_power(
                            suit, total_value, game_state.current_enemy, game_state.discard_pile,
                            game_state.tavern_deck, list(game_state.players), game_state, active_player
                        )

            damage = total_attack(cards_played, game_state.face_jester_played, enemy=game_state.current_enemy)
            game_state.current_enemy.apply_damage(damage)
            logger.info(
                f"{active_player.name} dealt {damage} damage to enemy {game_state.current_enemy.display_name} (total damage: {game_state.current_enemy.current_damage})")
            print(
                f"\033[93mDamage dealt to enemy: {damage}. (Total: {game_state.current_enemy.current_damage})\033[0m")

            if game_state.current_enemy.is_defeated():
                if exact_defeat(game_state.current_enemy.current_damage, game_state.current_enemy):
                    print("You have defeated the enemy EXACTLY! Place enemy on top of Tavern deck.")
                    logger.info(f"Enemy {game_state.current_enemy.display_name} defeated EXACTLY.")
                    game_state.tavern_deck.add_to_top(game_state.current_enemy)
                else:
                    print("Enemy defeated! Card goes to discard pile.")
                    logger.info(f"Enemy {game_state.current_enemy.display_name} defeated.")
                    game_state.discard_pile.add_to_top(game_state.current_enemy)
                for c in game_state.played_this_enemy:
                    game_state.discard_pile.add_to_top(c)
                game_state.played_this_enemy.clear()
                next_enemy = game_state.castle_deck.draw_one()
                logger.info(f"Next enemy: {next_enemy.display_name if next_enemy else 'None (game won)'}")
                game_state.current_enemy = next_enemy
                game_state.face_jester_played = False
                if next_enemy is not None:
                    print("\nA new enemy appears!\n")
                continue
            else:
                if not handle_enemy_attack(game_state, active_player):
                    logger.warning(f"{active_player.name} was defeated by enemy attack.")
                    return

            game_state.advance_turn()
            log_data_state(game_state)

        elif action == 'yield':
            logger.info(f"{active_player.name} yields.")
            logger.data(f"YIELD by {active_player.name}")
            if not handle_enemy_attack(game_state, active_player):
                logger.warning(f"{active_player.name} was defeated by enemy attack after yielding.")
                return
            game_state.advance_turn()
            log_data_state(game_state)

        else:
            show_invalid_action("Unknown action. Skipping turn for safety.")
            logger.warning(f"Unknown action for {active_player.name}: {action}")
            game_state.advance_turn()

        if game_state.check_loss():
            show_loss("A player could not satisfy enemy attack or no legal plays remain.")
            logger.error("Game lost: a player could not satisfy enemy attack or no legal plays remain.")
            return

    show_win()
    logger.info("Game won!")


def ask_player_count():
    print("How many players will playing? (2-4)")
    while True:
        try:
            num_players = int(input("> ").strip())
            if num_players not in (2, 3, 4):
                print("2, 3, or 4 players only.")
                logger.warning(f"Invalid player count input: {num_players}")
                continue
            break
        except ValueError as e:
            print("Please enter a number (2-4).")
            logger.warning(f"Non-integer player count input: {e}")
    return num_players


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        logger = get_logger("regicide.main")
        logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        print("\033[91mA fatal error occurred. See regicide.log for details.\033[0m")
