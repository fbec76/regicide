from typing import List

from game.card import Card


def valid_combo(cards):
    """
    Returns True if cards is a valid combo play; cards: list of Card.
    Combo: 2, 3, 4 cards; all same value; sum of values <= 10.
    Animal companions and Jesters cannot be in combos.
    """
    if len(cards) < 2 or len(cards) > 4:
        return False
    if any(c.c_type in ('animal', 'jester', 'enemy') for c in cards):
        return False
    value = cards[0].value
    if not all(c.value == value for c in cards):
        return False
    if sum(c.value for c in cards) > 10:
        return False
    return True

def can_play_jester(cards):
    """Returns True if a lone card is a Jester."""
    return len(cards) == 1 and cards[0].c_type == 'jester'

def valid_animal_pair(cards):
    """
    Returns True if cards is a valid animal pair: animal + one non-jester, or two animals.
    """
    if len(cards) == 2:
        if all(c.c_type == 'animal' for c in cards):
            return True
        if any(c.c_type == 'animal' for c in cards) and not any(c.c_type == 'jester' for c in cards):
            return True
    return False

def is_valid_play(cards):
    """
    Returns True if cards is a legal play for a turn.
    (Assumes enemy is not just defeated, and there are playable cards.)
    """
    if len(cards) == 1:
        # Any single card (not enemy face in Castle), including animal or jester
        return True
    if valid_combo(cards):
        return True
    if valid_animal_pair(cards):
        return True
    return False


def is_enemy_immune(enemy, card, jester_active=False):
    """
    Returns True if current enemy would block this card's suit effects.
    Jester negates immunity.
    """
    # Jester: disables suit powers immunity for all suits
    if jester_active:
        return False
    return enemy.suit == card.suit

def resolve_suit_power(suit, value, enemy, discard_pile, tavern_deck, players, game_state, acting_player):
    if suit == 'hearts':
        n = value
        if len(discard_pile) > 0:
            restored = discard_pile.shuffle_and_return_n_cards(n)
            for card_ in restored:
                tavern_deck.add_to_bottom(card_)
            print(f"♥ {len(restored)} cards healed to tavern deck.")
    elif suit == 'diamonds':
        to_draw = value
        drawn = 0
        start = players.index(acting_player)
        idx = 0
        n_players = len(players)
        while drawn < to_draw:
            player = players[(start + idx) % n_players]
            if player.hand_size() < player.hand_limit and not tavern_deck.is_empty():
                c = tavern_deck.draw_one()
                player.add_card(c)
                drawn += 1
            idx += 1
            if all(p.hand_size() == p.hand_limit for p in players) or tavern_deck.is_empty():
                break
        print(f"♦ {drawn} cards drawn among players.")
    elif suit == 'clubs':
        pass
    elif suit == 'spades':
        enemy.apply_shield(value)
        print(f"♠ Enemy shielded by {value} → Total: {enemy.shield}")


def total_attack(cards: List[Card], jester_active=False, enemy=None):
    """
    Computes attack value to enemy for cards played (handling combos, animal pair, clubs).
    If playing against clubs enemy with immunity (jester_active==False), no club doubling.
    Returns int: total damage.
    """
    damage = 0
    has_club = False
    for c in cards:
        if c.c_type == 'jester':
            continue
        damage += c.value
        if c.suit == 'clubs':
            has_club = True
    # Clubs doubling: Only if a club is played, and not against club enemy unless jester active
    if has_club and (not (enemy and enemy.suit == 'clubs' and not jester_active)):
        return 2 * damage
    else:
        return damage

def exact_defeat(attack_so_far, enemy):
    """
    True if the sum total of damage so far equals enemy's max_health, per "exact" defeat bonus.
    """
    return attack_so_far == enemy.max_health
