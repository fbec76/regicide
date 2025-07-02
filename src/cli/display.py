import sys

SUIT_SYMBOLS = {
    "hearts": "♥",
    "diamonds": "♦",
    "clubs": "♣",
    "spades": "♠",
    "joker": "Ⓙ"
}

SUIT_COLORS = {
    "hearts": "\033[91m",  # Red
    "diamonds": "\033[91m",  # Red
    "clubs": "\033[90m",  # Grey
    "spades": "\033[90m",  # Grey
    "joker": "\033[93m",  # Yellow
}
RESET_COLOR = "\033[0m"


def card_ascii(card, width=8, highlight=False):
    """
    Returns a list of strings for the card's ASCII art, with highlighting if specified.
    Handles all card types & centers attack|health for enemies for any reasonable width>=7.
    """

    s = SUIT_SYMBOLS.get(card.suit, "?")
    color = SUIT_COLORS.get(card.suit, "")
    r = str(card.rank)
    v = card.value
    hl = "\033[1m" if highlight else ""
    endhl = "\033[0m" if highlight else ""
    sym = f"{color}{s}{RESET_COLOR}" if sys.stdout.isatty() else s

    # Special handling/centering for center display
    if card.c_type == "animal":
        label = "A"
        center = f"{hl}A{endhl}".center(width - 2)
    elif card.c_type == "jester":
        label = "★"
        center = f"{color}★{RESET_COLOR}".center(width - 2)
    elif card.c_type == "enemy":
        label = r
        attack = card.attack
        health = card.max_health
        # Center the "hp|attack" string for arbitrary width
        ha = f"{health}|{attack}"
        center = f"{hl}{ha}{endhl}"
        center = center.center(width - len(str(center)))
    else:
        label = r
        center = f"{hl}{str(v)}{endhl}".center(width - 2)

    # Compose the lines:
    # Top label left, suit right
    pad_r = width - 3 - len(label)         # for left-aligned rank
    pad_l = width - 3 - len(label)         # for right-aligned rank
    # Guarantee >=1 space if width is narrow (should use at least width=7+!)
    pad_r = max(pad_r, 1)
    pad_l = max(pad_l, 1)

    top     = hl + f"┌{'─' * (width - 2)}┐" + endhl
    line_1  = hl + f"│{label}{' ' * pad_r}{sym}│" + endhl
    line_2  = hl + f"│{' ' * (width - 2)}│" + endhl
    line_3  = hl + f"│{center}│" + endhl
    line_4  = hl + f"│{' ' * (width - 2)}│" + endhl
    line_5  = hl + f"│{sym}{' ' * pad_l}{label}│" + endhl
    bot     = hl + f"└{'─' * (width - 2)}┘" + endhl
    return [top, line_1, line_2, line_3, line_4, line_5, bot]


def print_cards(cards, highlight_idxs=None):
    """Prints cards horizontally as ascii art. Optionally highlight given indexes."""
    if not cards:
        print("[empty]")
        return
    ascii_cards = [
        card_ascii(card, highlight=(highlight_idxs and i in highlight_idxs))
        for i, card in enumerate(cards)
    ]
    for rows in zip(*ascii_cards):
        print("  ".join(rows))


def print_separator(char="=", width=60):
    print(char * width)


def show_enemy(enemy):
    print_separator("=")
    print("                ENEMY ENCOUNTERED")
    print_separator("=")
    card_lines = card_ascii(enemy, width=15, highlight=True)
    for line in card_lines:
        print(line.center(60))
    print(
        f"Health: {enemy.max_health - enemy.current_damage}/{enemy.max_health}\t\tAttack: {enemy.effective_attack()} (base {enemy.attack}, shield {enemy.shield})")
    print(
        f"Suit: {enemy.suit.capitalize()}\t\tImmunity: {'OFF' if getattr(enemy, 'suit_immunity_cancelled', False) else 'ON'}")
    print_separator()


def show_players(players):
    print_separator("-")
    print("Players:")
    for p in players:
        ys = " (yielded)" if getattr(p, "has_yielded", False) else ""
        print(f"  {p.name}: hand ({len(p.hand)}/{p.hand_limit}){ys}")
    print_separator("-")


def show_player_hand(player):
    print_separator("-")
    print(f"{player.name}'s hand:")
    print_cards(player.hand)
    print("".join([f"[{i}]       " for i in range(len(player.hand))]))
    print_separator("-")


def show_tavern(tavern_deck):
    print(f"Tavern deck: {len(tavern_deck)} card(s) remaining.")


def show_discard(discard_pile):
    print(f"Discard pile: {len(discard_pile)} card(s).")


def show_play_area(play_area):
    if not play_area:
        print("(No cards played against this enemy yet.)")
        return
    print("Cards played against enemy so far:")
    print_cards(play_area)


def show_action_prompt(player):
    print_separator("-")
    print(f"{player.name}, it's your turn.")
    print("What would you like to do?")
    print("  1. Play a card/combo")
    print("  2. Yield (take damage instead of playing)")
    print("Enter [1] or [2].")
    print_separator("-")


def show_invalid_action(msg="Invalid action. Please try again."):
    print(f"\033[91m!! {msg}\033[0m")


def show_yield_forbidden():
    print("\033[93mYou cannot yield this turn. Play a card or combo.\033[0m")


def show_win():
    print_separator("=")
    print("\033[92m" + " " * 10 + "CONGRATULATIONS! YOU HAVE WON REGICIDE!\033[0m")
    print_separator("=")


def show_loss(reason=""):
    print_separator("=")
    print("\033[91m" + " " * 15 + "YOU HAVE BEEN DEFEATED!\033[0m")
    if reason:
        print("Reason:", reason)
    print_separator("=")


def show_discard_phase(player, damage_needed):
    print_separator()
    print(
        f"{player.name}, you must discard cards with a total value of at least {damage_needed} to cover the enemy's attack.")


def show_hearts_heal(count):
    print(f"Hearts Power: \033[91m{count}\033[0m card(s) healed from discard pile to tavern deck.")


def show_diamonds_draw(count, player):
    print(f"Diamonds Power: {player.name} draws {count} card(s) from the tavern deck.")


def show_clubs_double_damage(value):
    print(f"Clubs Power: Club card(s) deal double damage: {value} x2")


def show_spades_shield(value):
    print(f"Spades Power: Enemy attack is reduced by {value}.")


def show_jester_played():
    print("\033[93mJester played! Enemy suit power immunity is now OFF.\033[0m")


def show_played_cards(cards):
    print_separator("-")
    print("You played:")
    print_cards(cards)
    print_separator("-")


def show_current_state(game_state):
    print_separator("=")
    show_enemy(game_state.current_enemy)
    show_play_area(game_state.played_this_enemy)
    show_tavern(game_state.tavern_deck)
    show_discard(game_state.discard_pile)
    show_players(list(game_state.players))
    print_separator("=")
