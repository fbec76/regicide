from cli.display import show_player_hand


def prompt_action_choice(player):
    """Prompt player to choose action: play or yield."""
    while True:
        choice = input(f"\033[1m{player.name}\033[0m, choose action [1]Play, [2]Yield: ").strip()
        if choice == "1":
            return "play"
        if choice == "2":
            return "yield"
        print("\033[91mInvalid input. Enter 1 or 2.\033[0m")

def prompt_card_indexes(
    player,
    prompt_message,
    extra_validation_fn=None,
    validation_fail_message=None,
    discard_phase=False
):
    """Helper to prompt card index(es) with optional validation."""
    player_cards = player.hand
    while True:
        try:
            in_str = input(prompt_message).strip()
            if not in_str:
                if discard_phase:
                    print("\033[93mNo input provided. You must select cards to discard.\033[0m")
                    continue
                else:
                    return None
            indexes = [int(idx) for idx in in_str.split()]
            if any(idx < 0 or idx >= len(player_cards) for idx in indexes):
                print("\033[91mInvalid card index.\033[0m")
                continue
            unique_indexes = list(dict.fromkeys(indexes))
            cards = [player_cards[idx] for idx in unique_indexes]
            if extra_validation_fn and not extra_validation_fn(cards):
                print(f"\033[91m{validation_fail_message or 'Selected cards invalid.'}\033[0m")
                continue
            return cards
        except ValueError:
            print("\033[91mEnter space-separated valid indexes.\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            continue
    return None

def prompt_card_indexes_to_play(player):
    print(f"\033[1m{player.name}\033[0m, select card(s) to play:")
    print(" - One card")
    print(" - Valid combo (2-4 same value, max total 10)")
    print(" - Animal + one card (no jesters), or two Animals")
    print("Enter indexes separated by spaces (e.g., 0 3):")
    return prompt_card_indexes(player, "\033[92mEnter card indexes (space-separated, leave empty to yield):\033[0m ")

def prompt_card_indexes_to_discard(player, required_value):
    def enough(cards):
        total = sum(c.value for c in cards)
        print(f"\033[93mTotal discard value: {total} ({len(cards)} card{'s' if len(cards) != 1 else ''})\033[0m")
        return total >= required_value

    show_player_hand(player)
    return prompt_card_indexes(
        player,
        "\033[92mEnter discard indexes (no repeats):\033[0m ",
        extra_validation_fn=enough,
        validation_fail_message=f"Total value must be ≥ {required_value}.",
        discard_phase=True
    )

def prompt_next_player(players):
    print("\033[93mJester played! Choose next player:\033[0m")
    for i, p in enumerate(players):
        print(f" [{i}] {p.name} (Hand: {len(p.hand)})")
    while True:
        choice = input("\033[92mNext player number:\033[0m ")
        try:
            idx = int(choice)
            if 0 <= idx < len(players) and players[idx].is_alive():
                return players[idx]
            else:
                print("\033[91mInvalid choice.\033[0m")
        except ValueError:
            print("\033[91mEnter a valid number.\033[0m")