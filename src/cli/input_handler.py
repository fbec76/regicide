from .display import show_player_hand


def prompt_action_choice(player):
    """
    Prompt player to choose action for their turn.
    Returns 'play' or 'yield'
    """
    while True:
        choice = input(f"{player.name}, choose your action ([1] Play a card/combo, [2] Yield): ").strip()
        if choice == "1":
            return "play"
        if choice == "2":
            return "yield"
        print("Invalid input. Please enter 1 for play or 2 for yield.")

def _prompt_card_indexes(
    player,
    prompt_message,
    extra_validation_fn=None,
    validation_fail_message=None
):
    """
    General card selection helper.
    extra_validation_fn(cards) -> True if valid
    """
    player_cards = player.hand
    while True:
        try:
            in_str = input(prompt_message).strip()
            if not in_str:
                print("Please enter at least one index.")
                continue
            indexes = [int(idx) for idx in in_str.split()]
            if any(idx < 0 or idx >= len(player_cards) for idx in indexes):
                print("Invalid card index given.")
                continue
            unique_indexes = list(dict.fromkeys(indexes))  # remove duplicates
            cards = [player_cards[idx] for idx in unique_indexes]
            # Optional extra validation (e.g., sum, play rules, etc)
            if extra_validation_fn and not extra_validation_fn(cards):
                print(validation_fail_message or "Selected cards do not meet requirements.")
                continue
            return cards
        except ValueError:
            print("Please enter space-separated valid indexes.")
        except Exception as e:
            print("Error:", e)
            continue
    return None

def prompt_card_indexes_to_play(player):
    print(f"{player.name}, select the card(s) to play. You may:")
    print("  - Play one card")
    print("  - Play a valid combo (2-4 of same value, ≤10 total)")
    print("  - Play an Animal + one other card (no jesters), or two Animals")
    print("Enter card indexes separated by spaces (e.g., 0 3 for two cards):")
    return _prompt_card_indexes(player, "Indexes: ")

def prompt_card_indexes_to_discard(player, required_value):
    def enough(cards):
        total = sum(c.value for c in cards)
        print(f"Selected cards total value: {total}")
        return total >= required_value

    show_player_hand(player)
    print(f"You must discard cards to cover at least {required_value} points of damage.")
    print("Enter card indexes separated by spaces (no repeats allowed):")
    return _prompt_card_indexes(
        player,
        "Discard indexes: ",
        extra_validation_fn=enough,
        validation_fail_message=f"Total value insufficient. Need at least {required_value}."
    )

def prompt_next_player(players):
    """
    When a jester is played, prompt: choose who goes next from alive players.
    """
    print("Jester has been played! Who should go next?")
    for i, p in enumerate(players):
        print(f" [{i}] {p.name} (Hand: {len(p.hand)})")
    while True:
        choice = input("Enter number for next player: ")
        try:
            idx = int(choice)
            if 0 <= idx < len(players) and players[idx].is_alive():
                return players[idx]
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")

def prompt_continue():
    input("Press Enter to continue...")

def prompt_any_key(message="Press Enter to continue..."):
    input(message)