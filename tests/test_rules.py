from game.card import Card
from game.rules import valid_combo, can_play_jester, valid_animal_pair, is_valid_play, is_enemy_immune, total_attack, exact_defeat
from game.enemy import Enemy

# Helper cards
def make_card(suit, rank, value, c_type):
    return Card(suit, rank, value, c_type, f"{rank} of {suit}")

def test_valid_combo():
    c1 = make_card('hearts', '2', 2, 'number')
    c2 = make_card('spades', '2', 2, 'number')
    c3 = make_card('clubs', '2', 2, 'number')
    assert valid_combo([c1, c2])
    assert valid_combo([c1, c2, c3])
    c4 = make_card('diamonds', '2', 2, 'number')
    assert valid_combo([c1, c2, c3, c4])
    # Too many cards
    assert not valid_combo([c1, c2, c3, c4, c1])
    # Different values
    c5 = make_card('hearts', '3', 3, 'number')
    assert not valid_combo([c1, c5])
    # Animal or jester in combo
    animal = make_card('hearts', 'A', 1, 'animal')
    jester = make_card('joker', 'Jester', 0, 'jester')
    assert not valid_combo([c1, animal])
    assert not valid_combo([c1, jester])
    # Sum > 10
    c6 = make_card('hearts', '5', 5, 'number')
    assert not valid_combo([c6, c6, c6])

def test_can_play_jester():
    jester = make_card('joker', 'Jester', 0, 'jester')
    assert can_play_jester([jester])
    assert not can_play_jester([jester, jester])
    c1 = make_card('hearts', '2', 2, 'number')
    assert not can_play_jester([c1])

def test_valid_animal_pair():
    animal1 = make_card('hearts', 'A', 1, 'animal')
    animal2 = make_card('spades', 'A', 1, 'animal')
    c1 = make_card('hearts', '2', 2, 'number')
    jester = make_card('joker', 'Jester', 0, 'jester')
    assert valid_animal_pair([animal1, animal2])
    assert valid_animal_pair([animal1, c1])
    assert not valid_animal_pair([animal1, jester])
    assert not valid_animal_pair([animal1])
    assert not valid_animal_pair([animal1, animal2, c1])

def test_is_valid_play():
    animal = make_card('hearts', 'A', 1, 'animal')
    c1 = make_card('hearts', '2', 2, 'number')
    c2 = make_card('spades', '2', 2, 'number')
    jester = make_card('joker', 'Jester', 0, 'jester')
    # Single card
    assert is_valid_play([c1])
    # Combo
    assert is_valid_play([c1, c2])
    # Animal pair
    assert is_valid_play([animal, c1])
    # Jester
    assert is_valid_play([jester])
    # Invalid
    assert not is_valid_play([animal, jester])

def test_is_enemy_immune():
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    card = make_card('hearts', '2', 2, 'number')
    other = make_card('spades', '2', 2, 'number')
    # Immune to same suit
    assert is_enemy_immune(enemy, card)
    # Not immune to other suit
    assert not is_enemy_immune(enemy, other)
    # Jester disables immunity
    assert not is_enemy_immune(enemy, card, jester_active=True)

def test_total_attack():
    enemy = Enemy('clubs', 'J', 10, 'enemy', 'Jack of Clubs', 20, 10)
    c1 = make_card('clubs', '2', 2, 'number')
    c2 = make_card('hearts', '3', 3, 'number')
    # Club doubling, not immune
    assert total_attack([c1, c2], jester_active=True, enemy=enemy) == 10
    # Club doubling, immune
    assert total_attack([c1, c2], jester_active=False, enemy=enemy) == 5
    # No clubs
    assert total_attack([c2], enemy=enemy) == 3

def test_exact_defeat():
    enemy = Enemy('spades', 'K', 10, 'enemy', 'King of Spades', 40, 20)
    assert exact_defeat(40, enemy)
    assert not exact_defeat(39, enemy)

