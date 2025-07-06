from game.rules import total_attack, is_enemy_immune, is_valid_play, exact_defeat, resolve_suit_power, can_play_jester
from game.player import Player
from game.deck import TavernDeck, CastleDeck, DiscardPile
from game.enemy import Enemy
from game.game_state import GameState

# CLI
from cli.display import (
    show_current_state, show_player_hand, show_invalid_action, show_win, show_loss,
    show_yield_forbidden, show_discard_phase, show_played_cards, show_jester_played
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
        name = input(f"Enter name for player {i + 1}: ").strip() or f'Player{i + 1}'
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


#### MAIN GAME LOOP

def main():
    print("Welcome to Regicide (Terminal Edition)!")
    print("How many players? (2-4)")
    while True:
        try:
            num_players = int(input("> ").strip())
            if num_players not in (2, 3, 4):
                print("2, 3, or 4 players only.")
                continue
            break
        except ValueError:
            print("Please enter a number (2-4).")

    hand_size_table = {1: 8, 2: 7, 3: 6, 4: 5}
    max_hand_size = hand_size_table[num_players]

    players = ask_player_names(num_players, max_hand_size)

    jacks, queens, kings = create_enemy_faces()
    castle_deck = CastleDeck(jacks, queens, kings)
    tavern_deck = TavernDeck(num_players)
    discard_pile = DiscardPile()

    game_state = GameState(players=players, tavern_deck=tavern_deck, castle_deck=castle_deck, discard_pile=discard_pile)

    deal_initial_hands(game_state)

    while not game_state.is_game_won():
        active_player = game_state.get_active_player()
        show_current_state(game_state)
        show_player_hand(active_player)

        if game_state.can_yield(active_player):
            action = prompt_action_choice(active_player)
        else:
            show_yield_forbidden()
            action = 'play'

        if action == 'play':
            while True:
                cards_played = prompt_card_indexes_to_play(active_player)
                if is_valid_play(cards_played):
                    break
                else:
                    show_invalid_action("That is not a valid play. Try again.")

            if can_play_jester(cards_played):
                show_jester_played()
                game_state.face_jester_played = True
                game_state.update_enemy_immunity()
                next_player = prompt_next_player(list(game_state.players))
                while game_state.get_active_player() != next_player:
                    game_state.advance_turn()
                continue

            for c in cards_played:
                if c.suit == game_state.current_enemy.suit and game_state.enemy_immunity:
                    confirm = input(
                        f"You are playing a card of the enemy's suit ({c.suit}) while immunity is ON. Sure? (y/N): ").strip().lower()
                    if confirm != 'y':
                        show_invalid_action("Play cancelled by player due to immunity confirmation.")
                        [active_player.add_card(card) for card in cards_played]
                        cards_played.clear()
                        break
            if not cards_played:
                continue

            active_player.play_cards(cards_played)
            game_state.played_this_enemy.extend(cards_played)
            show_played_cards(cards_played)

            total_value = sum(c.value for c in cards_played)
            for suit in ('hearts', 'diamonds', 'clubs', 'spades'):
                if any(c.suit == suit for c in cards_played):
                    # Check immunity
                    any_card_of_suit = next(c for c in cards_played if c.suit == suit)
                    if not is_enemy_immune(game_state.current_enemy, any_card_of_suit, game_state.face_jester_played):
                        resolve_suit_power(
                            suit, total_value, game_state.current_enemy, game_state.discard_pile,
                            game_state.tavern_deck, list(game_state.players), game_state, active_player
                        )

            damage = total_attack(cards_played, game_state.face_jester_played, enemy=game_state.current_enemy)
            game_state.current_enemy.apply_damage(damage)
            print(
                f"\033[93mTotal damage dealt to enemy: {damage}. (Damage so far: {game_state.current_enemy.current_damage})\033[0m")

            # Step 4: Check defeat or suffer damage
            if game_state.current_enemy.is_defeated():
                if exact_defeat(game_state.current_enemy.current_damage, game_state.current_enemy):
                    print("You have defeated the enemy EXACTLY! Place enemy on top of Tavern deck.")
                    game_state.tavern_deck.add_to_top(game_state.current_enemy)
                else:
                    print("Enemy defeated! Card goes to discard pile.")
                    game_state.discard_pile.add_to_top(game_state.current_enemy)
                # Discard all played cards
                for c in game_state.played_this_enemy:
                    game_state.discard_pile.add_to_top(c)
                game_state.played_this_enemy.clear()
                # Next enemy
                next_enemy = game_state.castle_deck.draw_one()
                game_state.current_enemy = next_enemy
                game_state.face_jester_played = False
                # Skip enemy retaliation this turn, next player starts
                if next_enemy is not None:
                    print("\nA new enemy appears!\n")
                continue  # Next player's turn
            else:
                # Enemy attacks!
                if not handle_enemy_attack(game_state, active_player):
                    return

            # Advance turn
            game_state.advance_turn()

        elif action == 'yield':
            if not handle_enemy_attack(game_state, active_player):
                return
            # Advance turn
            game_state.advance_turn()

        else:
            show_invalid_action("Unknown action. Skipping turn for safety.")
            game_state.advance_turn()

        # End-of-turn/game checks
        if game_state.check_loss():
            show_loss("A player could not satisfy enemy attack or no legal plays remain.")
            return

    show_win()


if __name__ == "__main__":
    main()
