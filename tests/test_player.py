from game.player import Player
from game.card import Card

def make_card(suit='hearts', rank='2', value=2, c_type='number'):
    return Card(suit, rank, value, c_type, f"{rank} of {suit}")

def test_add_card_and_hand_limit():
    p = Player('Alice', hand_limit=2)
    c1 = make_card()
    c2 = make_card('spades')
    c3 = make_card('clubs')
    assert p.add_card(c1)
    assert p.add_card(c2)
    assert not p.add_card(c3)  # hand limit reached
    assert p.hand_size() == 2

def test_play_cards():
    p = Player('Bob', hand_limit=3)
    c1 = make_card()
    c2 = make_card('spades')
    p.add_card(c1)
    p.add_card(c2)
    assert p.play_cards([c1])
    assert c1 not in p.hand
    assert c2 in p.hand
    # Try to play a card not in hand
    c3 = make_card('clubs')
    assert not p.play_cards([c3])

def test_show_hand():
    p = Player('Carol', hand_limit=2)
    c1 = make_card()
    c2 = make_card('spades')
    p.add_card(c1)
    p.add_card(c2)
    hand_names = p.show_hand()
    assert f"2 of hearts" in hand_names
    assert f"2 of spades" in hand_names

def test_is_alive():
    p = Player('Dan', hand_limit=1)
    assert p.is_alive()
    p.alive = False
    assert not p.is_alive()

