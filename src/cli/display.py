import sys
import re

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


def visible_len(s):
    """Count the length of the string as seen in the terminal (ignore ANSI codes)."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


def manual_center(text, width):
    """Center text within width using spaces, based on visible characters (ignoring ANSI)."""
    text_len = visible_len(text)
    total_space = max(0, width - text_len)
    left = total_space // 2
    right = total_space - left
    return ' ' * left + text + ' ' * right


def card_ascii(card, width=9, highlight=False):
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
        center = manual_center(f"{hl}A{endhl}", width - 2)
    elif card.c_type == "jester":
        label = "★"
        center = manual_center(f"{color}★{RESET_COLOR}", width - 2)
    elif card.c_type == "enemy":
        label = r
        attack = card.attack
        health = card.max_health
        # Center the "hp|attack" string for arbitrary width
        ha = f"{health}|{attack}"
        center = f"{hl}{ha}{endhl}"
        center = manual_center(center, width - 2)
    else:
        label = r
        center = manual_center(f"{hl}{str(v)}{endhl}", width - 2)

    # Compose the lines:
    # Top label left, suit right
    pad_r = width - 3 - len(label)  # for left-aligned rank
    pad_l = width - 3 - len(label)  # for right-aligned rank
    # Guarantee >=1 space if width is narrow
    pad_r = max(pad_r, 1)
    pad_l = max(pad_l, 1)

    top = hl + f"┌{'─' * (width - 2)}┐" + endhl
    line_1 = hl + f"│{label}{' ' * pad_r}{sym}│" + endhl
    line_2 = hl + f"│{' ' * (width - 2)}│" + endhl
    line_3 = hl + f"│{center}│" + endhl
    line_4 = hl + f"│{' ' * (width - 2)}│" + endhl
    line_5 = hl + f"│{sym}{' ' * pad_l}{label}│" + endhl
    bot = hl + f"└{'─' * (width - 2)}┘" + endhl
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
    print(manual_center("\033[1;31m🔥 ENEMY ENCOUNTERED 🔥\033[0m", 60))
    print_separator("=")
    card_lines = card_ascii(enemy, width=9, highlight=True)
    for line in card_lines:
        print(manual_center(line, 60))
    hp = enemy.max_health - enemy.current_damage
    atk = enemy.effective_attack()
    suit_immunity = 'OFF' if getattr(enemy, 'suit_immunity_cancelled', False) else 'ON'
    print(f"\033[91m♥ {hp}/{enemy.max_health}\033[0m  \033[93m⚔ {atk}\033[0m (atk {enemy.attack}, shd {enemy.shield})")
    print(f"\033[94mSuit:\033[0m {enemy.suit.capitalize()}  \033[96mImmunity:\033[0m {suit_immunity}")
    print_separator()

def show_players(players):
    print_separator("-")
    print("\033[1mPlayers:\033[0m " + ", ".join(
        f"{p.name}{' (yielded)' if getattr(p, 'has_yielded', False) else ''} ({len(p.hand)}/{p.hand_limit})" for p in players))
    print_separator("-")

def show_player_hand(player):
    print_separator("-")
    print(f"\033[1m{player.name}'s hand:\033[0m")
    print_cards(player.hand)
    print("  ".join([f"   [{i}]   " for i in range(len(player.hand))]))
    print_separator("-")

def show_tavern(tavern_deck):
    print(f"\033[92mTavern deck:\033[0m {len(tavern_deck)} cards remaining.")

def show_discard(discard_pile):
    print(f"\033[93mDiscard pile:\033[0m {len(discard_pile)} cards.")

def show_play_area(play_area):
    if not play_area:
        print("\033[90m-- No enemy cards played yet --\033[0m")
        return
    print("\033[1mCards against enemy:\033[0m")
    print_cards(play_area)

def show_action_prompt(player):
    print_separator("-")
    prompt = f"\033[1m{player.name}'s turn.\033[0m Choose:\n  [1] Play card/combo\n  [2] Yield"
    print(prompt)
    print_separator("-")

def show_invalid_action(msg="Invalid action. Try again."):
    print(f"\033[91m❌ {msg}\033[0m")

def show_yield_forbidden():
    print("\033[93m⚠ Yield forbidden this turn. Play a card.\033[0m")

def show_win():
    print_separator("=")
    print(manual_center("\033[1;92m✨ CONGRATS! YOU WON REGICIDE! ✨\033[0m", 60))
    print_separator("=")

def show_loss(reason=""):
    print_separator("=")
    print(manual_center("\033[1;91m☠ YOU ARE DEFEATED ☠\033[0m", 60))
    if reason:
        print(f"\033[91mReason: {reason}\033[0m")
    print_separator("=")

def show_discard_phase(player, damage_needed):
    print_separator()
    print(f"\033[93m{player.name}, discard cards totalling ≥ {damage_needed} to block attack.\033[0m")

def show_hearts_heal(count):
    print(f"\033[91m♥ Hearts Power:\033[0m {count} cards healed from discard to tavern.")

def show_diamonds_draw(count, player):
    print(f"\033[96m♦ Diamonds Power:\033[0m {player.name} draws {count} card(s) from tavern.")

def show_clubs_double_damage(value):
    print(f"\033[90m♣ Clubs Power:\033[0m Double damage: {value} x2")

def show_spades_shield(value):
    print(f"\033[90m♠ Spades Power:\033[0m Attack reduced by {value}.")

def show_jester_played():
    print("\033[93m✨ Jester played! Enemy suit immunity OFF. ✨\033[0m")

def show_played_cards(cards):
    print_separator("-")
    print("\033[1mYou played:\033[0m")
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