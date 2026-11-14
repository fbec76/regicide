"""Preview handler tests (plan step B1 / Decision D2): three-state suit
power semantics, the defeats-enemy hint, zero state mutation, and the
active-client/phase/schema guard. Real engine via initialize_game, no wire
(mirrors test_bridge.py).

Deterministic setups: hand/enemy shapes needed for specific preview states
are found by scanning ``random.seed`` values (same proven policy as
test_bridge.py::test_bridge_jester_peek_inject_and_cleanup) — each
``initialize_game`` costs <1ms, and every scanned bridge is a fully
initialized, untouched game.
"""
import asyncio
import itertools

from test_bridge import of_type, snapshot

# 2 players: tavern 40 + castle 12 = 52 cards (mirrors conftest invariant).
FULL_CAMPAIGN_CARDS = 52

TWO_PLAYERS = {"c1": "Alice", "c2": "Bob"}


# ---------------------------------------------------------------------------
# Setup scanning helpers
# ---------------------------------------------------------------------------

def _hand_and_enemy(bridge):
    game_state = bridge.engine.game_state
    return game_state.get_active_player().hand, game_state.current_enemy


def _active_client(bridge):
    return bridge.client_of_player[
        bridge.engine.game_state.get_active_player()]


def _combo_indices(hand, suits, max_value):
    """Indices of one number card per suit, all with the same value
    <= max_value (a legal combo when 2 <= len(suits) <= 4). None if absent."""
    for value in range(2, max_value + 1):
        found = []
        for suit in suits:
            match = next((i for i, c in enumerate(hand)
                          if c.suit == suit and c.value == value
                          and c.c_type == "number"), None)
            if match is None:
                found = None
                break
            found.append(match)
        if found:
            return found
    return None


def _defeats_indices(hand):
    """Indices of a legal clubs+partner combo of two value-5 cards
    (sum 10, club doubles → 20 damage = a fresh Jack's max_health)."""
    club5 = next((i for i, c in enumerate(hand)
                  if c.suit == "clubs" and c.value == 5
                  and c.c_type == "number"), None)
    if club5 is None:
        return None
    partner = next((i for i, c in enumerate(hand)
                    if i != club5 and c.value == 5 and c.suit != "clubs"
                    and c.c_type == "number"), None)
    return [club5, partner] if partner is not None else None


def _spade_index(hand):
    return next((i for i, c in enumerate(hand) if c.suit == "spades"), None)


# name → predicate(hand, enemy) identifying one preview scenario
SETUP_SPECS = {
    # hearts+diamonds+clubs combo vs non-immune enemy: hearts/diamonds fire,
    # club doubling applies.
    "fires_triple": lambda hand, enemy: (
        enemy.suit == "spades"
        and _combo_indices(hand, ("hearts", "diamonds", "clubs"), 3)),
    # matching-suit enemy blocks the hearts power; diamonds still fires.
    "blocked_pair": lambda hand, enemy: (
        enemy.suit == "hearts"
        and _combo_indices(hand, ("hearts", "diamonds"), 5)),
    # clubs doubling is OFF vs a clubs enemy (mirrors total_attack).
    "clubs_off_pair": lambda hand, enemy: (
        enemy.suit == "clubs"
        and _combo_indices(hand, ("hearts", "clubs"), 5)),
    # spade vs spades-enemy immunity → deferred to pending_spades.
    "pending_spade": lambda hand, enemy: (
        enemy.suit == "spades" and _spade_index(hand) is not None),
    # clubs5 + another 5 vs a fresh non-clubs Jack: 20 damage == max_health
    # (doubling is off vs a clubs enemy, so exclude that suit).
    "defeats_pair": lambda hand, enemy: (
        enemy.max_health == 20 and enemy.suit != "clubs"
        and _defeats_indices(hand)),
}


def _scan_setups(make_bridge, names, max_seeds=4000):
    """Scan seeds until every named setup has an initialized bridge."""
    specs = {name: SETUP_SPECS[name] for name in names}
    found = {}
    for seed in range(max_seeds):
        bridge = make_bridge(TWO_PLAYERS, seed=seed)
        asyncio.run(bridge.initialize_game())
        hand, enemy = _hand_and_enemy(bridge)
        for name, predicate in specs.items():
            if name not in found and predicate(hand, enemy):
                found[name] = bridge
        if len(found) == len(specs):
            break
    missing = set(specs) - set(found)
    assert not missing, (
        f"seed scan exhausted ({max_seeds} seeds); missing setups: "
        f"{sorted(missing)}")
    return found


def _request_preview(bridge, transport, indices):
    """Send a preview from the active client and return its response.
    ``handle_player_action`` returns None for previews (no step) — the
    response is what the transport delivered to the requester."""
    client = _active_client(bridge)
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "preview", "card_indices": indices}))
    assert info is None  # a preview never steps
    previews = of_type(transport.to_client(client), "preview")
    assert previews, f"no preview response sent to {client}"
    return previews[-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_preview_valid_play_states(make_bridge, transport):
    setups = _scan_setups(
        make_bridge, ["fires_triple", "blocked_pair", "clubs_off_pair",
                      "pending_spade"])

    # Fires: hearts+diamonds+clubs combo vs a non-immune (spades) enemy.
    bridge = setups["fires_triple"]
    hand, enemy = _hand_and_enemy(bridge)
    indices = _combo_indices(hand, ("hearts", "diamonds", "clubs"), 3)
    value = hand[indices[0]].value
    preview = _request_preview(bridge, transport, indices)
    assert preview["type"] == "preview"
    assert preview["valid"] is True
    assert preview["damage"] == 2 * 3 * value  # combo sum, club-doubled
    assert preview["double_damage"] is True
    assert preview["powers"] == {"hearts": "fires", "diamonds": "fires"}
    assert preview["defeats_enemy"] is False
    assert preview["yield_attack"] == enemy.effective_attack()

    # Blocked: matching-suit enemy suppresses the hearts power, but the
    # hearts damage still applies; diamonds fires.
    bridge = setups["blocked_pair"]
    hand, _ = _hand_and_enemy(bridge)
    indices = _combo_indices(hand, ("hearts", "diamonds"), 5)
    value = hand[indices[0]].value
    preview = _request_preview(bridge, transport, indices)
    assert preview["valid"] is True
    assert preview["powers"] == {"hearts": "blocked", "diamonds": "fires"}
    assert preview["damage"] == 2 * value
    assert preview["double_damage"] is False

    # Clubs doubling OFF vs a clubs enemy: damage undoubled, hearts fires,
    # no "clubs" power entry (clubs = the doubling marker only).
    bridge = setups["clubs_off_pair"]
    hand, _ = _hand_and_enemy(bridge)
    indices = _combo_indices(hand, ("hearts", "clubs"), 5)
    value = hand[indices[0]].value
    preview = _request_preview(bridge, transport, indices)
    assert preview["valid"] is True
    assert preview["powers"] == {"hearts": "fires"}
    assert preview["damage"] == 2 * value
    assert preview["double_damage"] is False

    # Pending: spade vs a spades enemy (immunity on) — deferred to
    # pending_spades, shield lands once a Jester negates immunity.
    bridge = setups["pending_spade"]
    hand, _ = _hand_and_enemy(bridge)
    index = _spade_index(hand)
    card = hand[index]
    preview = _request_preview(bridge, transport, [index])
    assert preview["valid"] is True
    assert preview["powers"] == {"spades": "pending"}
    assert preview["damage"] == card.value  # no club → no doubling
    assert preview["double_damage"] is False


def test_preview_defeats_enemy_hint(make_bridge, transport):
    bridge = _scan_setups(make_bridge, ["defeats_pair"])["defeats_pair"]
    hand, enemy = _hand_and_enemy(bridge)
    indices = _defeats_indices(hand)

    # 20 damage >= the fresh Jack's 20 max_health → hint fires.
    preview = _request_preview(bridge, transport, indices)
    assert preview["valid"] is True
    assert preview["damage"] == 20
    assert preview["defeats_enemy"] is True

    # Negative hint: a single card can never defeat a fresh enemy.
    preview = _request_preview(bridge, transport, [0])
    assert preview["valid"] is True
    assert preview["damage"] < enemy.max_health
    assert preview["defeats_enemy"] is False


def test_preview_zero_mutation(make_bridge, transport, total_cards):
    bridge = _scan_setups(make_bridge, ["fires_triple"])["fires_triple"]
    game_state = bridge.engine.game_state
    client = _active_client(bridge)
    other = next(cid for cid in TWO_PLAYERS if cid != client)
    hand, _ = _hand_and_enemy(bridge)
    transport.clear()
    baseline = snapshot(game_state)

    # Valid preview: response to the requester only, state byte-identical.
    indices = _combo_indices(hand, ("hearts", "diamonds", "clubs"), 3)
    preview = _request_preview(bridge, transport, indices)
    assert preview["valid"] is True
    assert of_type(transport.to_client(client), "preview") == [preview]
    assert of_type(transport.to_client(other), "preview") == []
    assert snapshot(game_state) == baseline
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
    assert bridge.engine.done is False

    # Invalid play selection (3 cards not all the same value) → preview
    # {valid: false, reason}, still zero mutation.
    trio = next(combo for combo in itertools.combinations(range(len(hand)), 3)
                if len({hand[i].value for i in combo}) > 1)
    response = _request_preview(bridge, transport, list(trio))
    assert response["type"] == "preview"
    assert response["valid"] is False
    assert response["reason"]
    assert snapshot(game_state) == baseline

    # Malformed/empty selections → error, zero mutation, requester only.
    for bad in ([-1], [0, 0], ["0"], None, []):
        transport.clear()
        info = asyncio.run(bridge.handle_player_action(
            client, {"type": "preview", "card_indices": bad}))
        assert info is None
        assert of_type(transport.to_client(client), "error")
        assert of_type(transport.to_client(other), "error") == []
        assert snapshot(game_state) == baseline
        assert bridge.engine.done is False


def test_preview_guard(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state
    active = game_state.get_active_player()
    client = bridge.client_of_player[active]
    other = next(cid for cid in TWO_PLAYERS if cid != client)
    transport.clear()
    baseline = snapshot(game_state)

    # Non-active client → error, no state change.
    info = asyncio.run(bridge.handle_player_action(
        other, {"type": "preview", "card_indices": [0]}))
    assert info is None
    errors = of_type(transport.to_client(other), "error")
    assert errors and "turn" in errors[-1]["message"].lower()
    assert snapshot(game_state) == baseline
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS

    # Drive to the defense phase, then preview → not applicable.
    pick = next(i for i, c in enumerate(active.hand)
                if c.suit != "spades"
                or game_state.current_enemy.suit == "spades")
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": [pick]}))
    assert info["result"] == "defense_required"
    defense_snapshot = snapshot(game_state)
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "preview", "card_indices": [0]}))
    assert info is None
    errors = of_type(transport.to_client(client), "error")
    assert errors and "phase" in errors[-1]["message"].lower()
    assert snapshot(game_state) == defense_snapshot
    assert bridge.engine.done is False
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
