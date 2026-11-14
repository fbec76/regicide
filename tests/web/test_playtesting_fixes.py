"""Playtesting fixes: preview post-play attack, castle progress in state,
and the uncoverable-defense auto-loss. Real engine via initialize_game, no
wire (mirrors test_preview.py / test_bridge.py).

Note on the spec's "blocked spade → unchanged" case: a blocked spade is
structurally impossible — a spade card is immune only against a spades
enemy, and that exact combination routes to the pending branch (test_preview
powers semantics). The pending case below therefore covers the "spade does
not lower the attack immediately" behavior.
"""
import asyncio

from test_bridge import FULL_CAMPAIGN_CARDS, of_type
from test_preview import (_active_client, _defeats_indices, _hand_and_enemy,
                          _request_preview, _scan_setups, _spade_index,
                          TWO_PLAYERS)

# ---------------------------------------------------------------------------
# Setup scanning helpers
# ---------------------------------------------------------------------------

SETUP_SPECS = {
    # spade vs a non-spades enemy: the spade power fires (no immunity).
    "spade_fires": lambda hand, enemy: (
        enemy.suit != "spades" and _spade_index(hand) is not None),
    # spade vs a spades enemy: immunity defers the shield to pending.
    "pending_spade": lambda hand, enemy: (
        enemy.suit == "spades" and _spade_index(hand) is not None),
}


def _uncoverable_setup(bridge):
    """Indices shaping an uncoverable defense: a hearts/clubs number card to
    play (no diamonds hand-refill, no spade shield) plus a kept card whose
    value is below the enemy's attack. Returns (play_index, kept_card) or
    None if the seed's shape does not allow it."""
    game_state = bridge.engine.game_state
    enemy = game_state.current_enemy
    hand = game_state.get_active_player().hand
    play_index = next(
        (i for i, c in enumerate(hand)
         if c.c_type == "number" and c.suit in ("hearts", "clubs")), None)
    if play_index is None:
        return None
    kept = min((c for i, c in enumerate(hand) if i != play_index),
               key=lambda c: c.value)
    if kept.value >= enemy.attack:
        return None
    return play_index, kept


def _scan_one(make_bridge, predicate, max_seeds=4000):
    """First seed whose initialized bridge satisfies predicate."""
    for seed in range(max_seeds):
        bridge = make_bridge(TWO_PLAYERS, seed=seed)
        asyncio.run(bridge.initialize_game())
        if predicate(bridge):
            return bridge
    raise AssertionError(
        f"seed scan exhausted ({max_seeds} seeds); no matching setup")


def _scan_specs(make_bridge, names, max_seeds=4000):
    """Seed scan over the module-local SETUP_SPECS (test_preview._scan_setups
    is bound to that module's own spec table)."""
    found = {}
    for seed in range(max_seeds):
        bridge = make_bridge(TWO_PLAYERS, seed=seed)
        asyncio.run(bridge.initialize_game())
        hand, enemy = _hand_and_enemy(bridge)
        for name in names:
            if name not in found and SETUP_SPECS[name](hand, enemy):
                found[name] = bridge
        if len(found) == len(names):
            return found
    missing = set(names) - set(found)
    raise AssertionError(
        f"seed scan exhausted ({max_seeds} seeds); missing setups: "
        f"{sorted(missing)}")


# ---------------------------------------------------------------------------
# Fix 1 — preview post_play_attack / pending_shield_value
# ---------------------------------------------------------------------------

def test_preview_post_play_attack_spade_fires(make_bridge, transport):
    setups = _scan_specs(make_bridge, ["spade_fires", "pending_spade"])

    # Firing spade: the shield lands immediately, floored at 0.
    bridge = setups["spade_fires"]
    hand, enemy = _hand_and_enemy(bridge)
    index = _spade_index(hand)
    value = hand[index].value
    preview = _request_preview(bridge, transport, [index])
    assert preview["valid"] is True
    assert preview["powers"] == {"spades": "fires"}
    assert preview["post_play_attack"] == max(
        enemy.effective_attack() - value, 0)
    assert preview["pending_shield_value"] == 0

    # Pending spade (spade vs spades enemy): no immediate change; the
    # deferred shield value is reported so the client can note the
    # post-Jester effect.
    bridge = setups["pending_spade"]
    hand, enemy = _hand_and_enemy(bridge)
    index = _spade_index(hand)
    value = hand[index].value
    preview = _request_preview(bridge, transport, [index])
    assert preview["valid"] is True
    assert preview["powers"] == {"spades": "pending"}
    assert preview["post_play_attack"] == enemy.effective_attack()
    assert preview["pending_shield_value"] == value


def test_preview_post_play_attack_no_spade(make_bridge, transport):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    hand, enemy = _hand_and_enemy(bridge)
    index = next(i for i, c in enumerate(hand)
                 if c.suit != "spades" and c.c_type == "number")
    preview = _request_preview(bridge, transport, [index])
    assert preview["valid"] is True
    assert preview["post_play_attack"] == enemy.effective_attack()
    assert preview["pending_shield_value"] == 0


# ---------------------------------------------------------------------------
# Fix 2 — castle progress in the shared state
# ---------------------------------------------------------------------------

def test_state_includes_castle_progress(make_bridge, transport):
    bridge = _scan_setups(make_bridge, ["defeats_pair"])["defeats_pair"]
    game_state = bridge.engine.game_state
    client = _active_client(bridge)

    state = of_type(transport.to_client(client), "state")[-1]
    assert state["castle_defeated"] == 0
    assert state["castle_total"] == 12
    assert state["castle_remaining"] == 11

    # Defeat the fresh Jack exactly (club5 + partner5 = 10 doubled to 20).
    hand, _ = _hand_and_enemy(bridge)
    indices = _defeats_indices(hand)
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": indices}))
    assert info["result"] == "enemy_defeated"

    state = of_type(transport.to_client(client), "state")[-1]
    assert state["castle_defeated"] == 1
    assert state["castle_total"] == 12


# ---------------------------------------------------------------------------
# Fix 3 — uncoverable defense terminates the loss instead of hanging
# ---------------------------------------------------------------------------

def test_uncoverable_defense_auto_loss(make_bridge, transport, total_cards):
    bridge = _scan_one(make_bridge, _uncoverable_setup)
    game_state = bridge.engine.game_state
    active = game_state.get_active_player()
    client = bridge.client_of_player[active]
    play_index, kept = _uncoverable_setup(bridge)

    # Drain the hand via direct access (same policy as the jester setup
    # scan in test_bridge.py): move every card except the play card and
    # the kept low card into the discard pile — conservation-safe.
    play_card = active.hand[play_index]
    drained = [c for c in active.hand if c is not play_card and c is not kept]
    for c in drained:
        game_state.discard_pile.add_to_top(c)
    active.hand = [play_card, kept]

    transport.clear()
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": [0]}))

    # The play itself resolves (damage < max_health → defense phase); the
    # defense request then detects the uncoverable hand and forces the
    # empty defend, which defense_validation turns into the loss.
    assert info["result"] == "defense_required"
    assert bridge.engine.done is True
    assert bridge.game_over is True
    ended = of_type(transport.broadcasts, "game_ended")
    assert ended and ended[-1]["result"] == "player_defeated"

    # NO action_required was sent for the unplayable defense — to anyone.
    for cid in bridge.client_by_id:
        assert of_type(transport.to_client(cid), "action_required") == []

    # Conservation intact and the session is terminal.
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
    result = asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": [0]}))
    assert result is None
    errors = of_type(transport.to_client(client), "error")
    assert errors and errors[-1]["message"] == "The game is not running"
