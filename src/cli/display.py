import sys
import re

from game.card import Card

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
    "T": "\033[92m",  # Green for Tavern
    "D": "\033[95m",  # Purple for Discard
}
RESET_COLOR = "\033[0m"

MAX_WIDTH = 75


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

    s = SUIT_SYMBOLS.get(card.suit, card.suit)
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


def print_separator(char="=", width=MAX_WIDTH):
    print(char * width)


def show_enemy(enemy, game_state):
    print_separator("=")
    tavern_card_lines = card_ascii(
        Card(suit="T", rank="T", c_type="int", value=len(game_state.tavern_deck), display_name="Tavern"), width=9,
        highlight=True)
    discard_card_lines = card_ascii(
        Card(suit="D", rank="D", c_type="int", value=len(game_state.discard_pile), display_name="Discard"),
        width=9, highlight=True)
    enemy_card_lines = card_ascii(enemy, width=9, highlight=True)
    hp = enemy.max_health - enemy.current_damage
    atk = enemy.effective_attack()
    suit_immunity = 'ON' if game_state.enemy_immunity else 'OFF'

    # Prepare enemy info lines
    enemy_info_lines = []
    for index, line in enumerate(enemy_card_lines):
        card_part = line
        if index == 1:
            status_part = f"  \033[91m♥ {hp}/{enemy.max_health}\033[0m"
        elif index == 3:
            status_part = f"  \033[93m⚔ {atk}\033[0m"
        elif index == 5:
            status_part = f"  \033[96m⛨\033[0m {suit_immunity}"
        else:
            status_part = ""
        enemy_info_lines.append(card_part + status_part)

    # Pad enemy_info_lines to match the number of card lines if needed
    while len(enemy_info_lines) < len(enemy_card_lines):
        enemy_info_lines.append("")

    # Combine tavern and discard pile cards vertically
    right_lines = []
    for t_line, d_line in zip(tavern_card_lines, discard_card_lines):
        right_lines.append(f"{t_line}   {d_line}")

    # Print side by side
    for l, r in zip(enemy_info_lines, right_lines):
        remaining = MAX_WIDTH - visible_len(l + r)
        print(f"{l}{' ' * remaining}{r}")
    print_separator()


def show_players(players):
    print_separator("-")
    print("\033[1mPlayers:\033[0m " + ", ".join(
        f"{p.name}{' (yielded)' if p.has_yielded else ''} ({len(p.hand)}/{p.hand_limit})" for p in
        players))


def show_player_hand(player):
    print_separator("-")
    print(f"\033[1m{player.name}'s hand:\033[0m")
    print_cards(player.hand)
    print("  ".join([f"   [{i}]   " for i in range(len(player.hand))]))
    print_separator("-")


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
    print(manual_center("\033[1;92m✨ CONGRATS! YOU WON REGICIDE! ✨\033[0m", MAX_WIDTH))
    print_separator("=")


def show_loss(reason=""):
    print_separator("=")
    print(manual_center("\033[1;91m☠ YOU ARE DEFEATED ☠\033[0m", MAX_WIDTH))
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
    show_enemy(game_state.current_enemy, game_state)
    show_play_area(game_state.played_this_enemy)
    show_players(list(game_state.players))


def show_welcome():
    print("\033[1;96m")
    print(manual_center(r" _______  _______  _______ _________ _______ _________ ______   _______ ", MAX_WIDTH))
    print(manual_center(r" (  ____ )(  ____ \(  ____ \\__   __/(  ____ \\__   __/(  __  \ (  ____ \ ", MAX_WIDTH))
    print(manual_center(r" | (    )|| (    \/| (    \/   ) (   | (    \/   ) (   | (  \  )| (    \/ ", MAX_WIDTH))
    print(manual_center(r" | (____)|| (__    | |         | |   | |         | |   | |   ) || (__     ", MAX_WIDTH))
    print(manual_center(r" |     __)|  __)   | | ____    | |   | |         | |   | |   | ||  __)    ", MAX_WIDTH))
    print(manual_center(r" | (\ (   | (      | | \_  )   | |   | |         | |   | |   ) || (       ", MAX_WIDTH))
    print(manual_center(r" | ) \ \__| (____/\| (___) |___) (___| (____/\___) (___| (__/  )| (____/\ ", MAX_WIDTH))
    print(manual_center(r" |/   \__/(_______/(_______)\_______/(_______/\_______/(______/ (_______/ ", MAX_WIDTH))
    print("\033[0m")
    print(manual_center("\033[1;92mTerminal Edition\033[0m", MAX_WIDTH))
    print("\033[1;90mhttps://www.regicidegame.com/site_files/33132/upload_files/RegicideRulesA4.pdf\033[0m")
    print()
    print()
