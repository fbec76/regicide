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


def resolve_suit_power(card, enemy, discard_pile, tavern_deck, players, game_state, acting_player):
    """
    Applies the effect of the card's suit power PER THE RULES.
    This should be called after a legal play, and after checking immunity.

    enemy: Enemy object (current)
    discard_pile: DiscardPile/Deck
    tavern_deck: TavernDeck/Deck
    players: list of Player objects in turn order (starting from acting player)
    game_state: the current GameState (for extra state, e.g., shields)
    acting_player: Player object
    """
    if card.suit == 'hearts':
        # Hearts: Heal 'value' cards from discard back under tavern deck
        n = card.value
        if len(discard_pile) > 0:
            restored = discard_pile.shuffle_and_return_n_cards(n)
            for card_ in restored:
                tavern_deck.add_to_bottom(card_)
            print(f"Healed {len(restored)} cards from discard into tavern deck.")
    elif card.suit == 'diamonds':
        # Diamonds: Draw 'value' cards, one per player in order, skipping full hands
        to_draw = card.value
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
            # If all are full or tavern empty, stop
            if idx > (n_players * 2):
                break
        print(f"Diamond effect: {drawn} cards drawn among players.")
    elif card.suit == 'clubs':
        # Clubs: Damage is doubled in main attack calculation (handled in main engine, not here).
        pass
    elif card.suit == 'spades':
        # Spades: Reduces current enemy's attack value by this card's value
        enemy.apply_shield(card.value)
        print(f"Spade effect: Enemy shielded by {card.value} (now {enemy.shield})")


def total_attack(cards, jester_active=False, enemy=None):
    """
    Computes attack value to enemy for cards played (handling combos, animal pair, clubs).
    If playing against clubs enemy with immunity (jester_active==False), no club doubling.
    Returns int: total damage.
    """
    damage = 0
    club_count = 0
    for c in cards:
        if c.c_type == 'jester':
            continue
        elif c.suit == 'clubs':
            club_count += c.value
        else:
            damage += c.value
    # Clubs doubling: Disabled against club enemy unless jester active
    if enemy and enemy.suit == 'clubs' and not jester_active:
        damage += club_count  # Just normal
    else:
        damage += 2 * club_count
    return damage


def yield_allowed(game_state, player):
    """
    Check if player is allowed to yield this turn (not all others have yielded last turn).
    """
    return game_state.can_yield(player)


def must_handle_enemy_attack(enemy, player):
    """Returns enemy's attack after shields."""
    return enemy.effective_attack()


def exact_defeat(attack_so_far, enemy):
    """
    True if the sum total of damage so far equals enemy's max_health, per "exact" defeat bonus.
    """
    return attack_so_far == enemy.max_health
