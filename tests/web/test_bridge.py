"""Handler-level bridge tests: WebSocketCLIBridge constructed directly with
fake broadcast/send callbacks and the real Engine (via initialize_game).
No wire, no websockets — handlers are driven with plain asyncio.run().
"""
import asyncio

from game.rules import total_attack

# 2 players: tavern 40 + castle 12 = 52 cards (mirrors conftest invariant).
FULL_CAMPAIGN_CARDS = 52

TWO_PLAYERS = {"c1": "Alice", "c2": "Bob"}
THREE_PLAYERS = {"c1": "Alice", "c2": "Bob", "c3": "Carol"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def snapshot(game_state):
    """Full read-only fingerprint of the game state for byte-identical
    before/after comparisons on rejected actions."""
    enemy = game_state.current_enemy
    return {
        "hands": {p.name: [c.display_name for c in p.hand] for p in game_state.players},
        "tavern": [c.display_name for c in game_state.tavern_deck.cards],
        "castle": [c.display_name for c in game_state.castle_deck.cards],
        "discard": [c.display_name for c in game_state.discard_pile.cards],
        "enemy": None if enemy is None else {
            "name": enemy.display_name,
            "damage": enemy.current_damage,
            "shield": enemy.shield,
        },
        "active": game_state.get_active_player().name,
        "against": [c.display_name for c in game_state.cards_against_enemy],
        "pending_spades": [c.display_name for c in game_state.pending_spades],
        "yields": dict(game_state.yields_in_a_row),
        "total_turns": game_state.total_turns,
        "face_jester_played": game_state.face_jester_played,
        "enemy_immunity": game_state.enemy_immunity,
    }


def cover_indices(hand, needed):
    """Greedy discard pick: highest values first until needed is covered.
    If the hand cannot cover, every index is returned (the engine then
    declares the loss via defense_validation)."""
    order = sorted(range(len(hand)), key=lambda i: hand[i].value, reverse=True)
    picked, total = [], 0
    for i in order:
        if total >= needed:
            break
        picked.append(i)
        total += hand[i].value
    return picked


def of_type(messages, message_type):
    return [m for m in messages if m["type"] == message_type]


def last_action_required(transport, client_id):
    reqs = of_type(transport.to_client(client_id), "action_required")
    assert reqs, f"no action_required sent to {client_id}"
    return reqs[-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bridge_initialize_and_state_sent(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state

    for client_id, name in TWO_PLAYERS.items():
        states = of_type(transport.to_client(client_id), "state")
        assert len(states) == 1
        state = states[0]
        assert state["you"]["name"] == name
        assert state["you"]["hand_size"] == 7
        assert len(state["you"]["hand"]) == 7
        assert {"index", "suit", "rank", "value", "display_name", "c_type"} <= set(
            state["you"]["hand"][0])
        assert state["opponents"] == [
            {"name": other, "hand_size": 7}
            for cid, other in TWO_PLAYERS.items() if cid != client_id
        ]
        assert state["enemy"]["max_health"] in (20, 30, 40)
        assert "effective_attack" in state["enemy"]
        # 12 castle enemies minus the one already drawn as current_enemy.
        assert state["castle_remaining"] == 11
        assert state["phase"] == "attack"

    active = game_state.get_active_player()
    active_client = bridge.client_of_player[active]
    other_client = next(cid for cid in TWO_PLAYERS if cid != active_client)

    # Opening action request goes to the active player's client only.
    reqs = of_type(transport.to_client(active_client), "action_required")
    assert reqs == [{"type": "action_required", "phase": "attack",
                     "player": active.name, "can_yield": True}]
    assert of_type(transport.to_client(other_client), "action_required") == []

    # Active player's client_id is revealed only to its owner.
    owner_state = of_type(transport.to_client(active_client), "state")[0]
    assert owner_state["active_player_client_id"] == active_client
    other_state = of_type(transport.to_client(other_client), "state")[0]
    assert "active_player_client_id" not in other_state
    assert other_state["active_player"] == active.name

    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS


def test_bridge_play_valid_and_invalid(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state
    active = game_state.get_active_player()
    active_client = bridge.client_of_player[active]
    other_client = next(cid for cid in TWO_PLAYERS if cid != active_client)
    baseline = snapshot(game_state)

    # Invalid: out-of-range index → error to sender, state byte-identical.
    info = asyncio.run(bridge.handle_player_action(
        active_client, {"type": "play", "card_indices": [99]}))
    assert info is None
    assert snapshot(game_state) == baseline
    assert bridge.engine.done is False
    assert of_type(transport.to_client(active_client), "error")

    # Invalid: wrong-typed indices (string, bool) → error, no state change.
    for bad in (["0"], [True]):
        info = asyncio.run(bridge.handle_player_action(
            active_client, {"type": "play", "card_indices": bad}))
        assert info is None
        assert snapshot(game_state) == baseline

    # Invalid: duplicates → error, no state change.
    info = asyncio.run(bridge.handle_player_action(
        active_client, {"type": "play", "card_indices": [0, 0]}))
    assert info is None
    assert snapshot(game_state) == baseline
    assert bridge.engine.done is False

    # Errors went to the sender only.
    assert of_type(transport.to_client(other_client), "error") == []

    # Valid: single-card play resolves to the exact Card and steps once.
    played = active.hand[0]
    info = asyncio.run(bridge.handle_player_action(
        active_client, {"type": "play", "card_indices": [0]}))
    assert info is not None
    assert info["result"] in ("defense_required", "turn_advanced")
    assert played not in active.hand
    assert game_state.current_enemy.current_damage == total_attack(
        [played], game_state.face_jester_played, enemy=game_state.current_enemy)
    assert bridge.engine.done is False
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS

    if info["result"] == "defense_required":
        assert last_action_required(transport, active_client)["phase"] == "defense"
    else:
        next_client = bridge.client_of_player[game_state.get_active_player()]
        assert last_action_required(transport, next_client)["phase"] == "attack"


def test_bridge_defense_flow(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state
    active = game_state.get_active_player()
    active_client = bridge.client_of_player[active]
    enemy = game_state.current_enemy

    # Play a card that cannot shield this enemy → attack stays > 0.
    pick = next(i for i, c in enumerate(active.hand)
                if c.suit != "spades" or enemy.suit == "spades")
    played = active.hand[pick]
    info = asyncio.run(bridge.handle_player_action(
        active_client, {"type": "play", "card_indices": [pick]}))
    assert info["result"] == "defense_required"
    assert enemy.effective_attack() > 0

    # The SAME client is asked to defend, with the needed total.
    request = last_action_required(transport, active_client)
    assert request == {"type": "action_required", "phase": "defense",
                       "player": active.name, "needed": enemy.effective_attack()}
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS

    needed = enemy.effective_attack()
    indices = cover_indices(active.hand, needed)
    assert sum(active.hand[i].value for i in indices) >= needed
    discarded = [active.hand[i] for i in indices]
    info = asyncio.run(bridge.handle_player_action(
        active_client, {"type": "defend", "card_indices": indices}))
    assert info["result"] == "defense_success"
    assert all(c not in active.hand for c in discarded)
    assert bridge.engine.phase == "attack"
    assert game_state.get_active_player() is not active

    resolved = of_type(transport.broadcasts, "defense_resolved")
    assert resolved and resolved[-1]["player"] == active.name
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS


def test_bridge_yield_forbidden_error(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state
    player_a = game_state.get_active_player()
    client_a = bridge.client_of_player[player_a]
    client_b = next(cid for cid in TWO_PLAYERS if cid != client_a)

    # A yields: the enemy attack keeps the defense phase with A — yield
    # does not dodge an attack.
    info = asyncio.run(bridge.handle_player_action(
        client_a, {"type": "play", "card_indices": []}))
    assert info["result"] == "yielded_defense_required"
    assert last_action_required(transport, client_a)["phase"] == "defense"
    indices = cover_indices(player_a.hand, game_state.current_enemy.effective_attack())
    info = asyncio.run(bridge.handle_player_action(
        client_a, {"type": "defend", "card_indices": indices}))
    assert info["result"] == "defense_success"
    assert game_state.get_active_player() is not player_a

    # B attempts to yield: forbidden (A already yielded on the previous
    # turn) → engine-level rejection, no state change.
    player_b = game_state.get_active_player()
    assert player_b is not player_a
    baseline = snapshot(game_state)
    info = asyncio.run(bridge.handle_player_action(
        client_b, {"type": "play", "card_indices": []}))
    assert info is not None and info["error"] == "Yield forbidden"
    rejected = of_type(transport.to_client(client_b), "action_rejected")
    assert rejected and "yield" in rejected[-1]["error"].lower()
    assert snapshot(game_state) == baseline
    assert game_state.get_active_player() is player_b
    assert bridge.engine.done is False
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS


def test_bridge_jester_peek_inject_and_cleanup(make_bridge, transport, total_cards):
    # 3 players: the tavern deck holds exactly one Jester. Find a seed where
    # the opening active player holds it.
    bridge = None
    for seed in range(300):
        candidate = make_bridge(THREE_PLAYERS, seed=seed)
        asyncio.run(candidate.initialize_game())
        hand = candidate.engine.game_state.get_active_player().hand
        if any(c.c_type == "jester" for c in hand):
            bridge = candidate
            transport.clear()
            break
    assert bridge is not None, "no seed put a Jester in the active player's hand"

    async def scenario():
        game_state = bridge.engine.game_state
        active = game_state.get_active_player()
        active_client = bridge.client_of_player[active]
        jester_index = next(
            i for i, c in enumerate(active.hand) if c.c_type == "jester")
        baseline = snapshot(game_state)

        # Play the Jester: the bridge peeks (no step yet) and asks the
        # ACTING client only for the choice.
        task = asyncio.ensure_future(bridge.handle_player_action(
            active_client, {"type": "play", "card_indices": [jester_index]}))
        await asyncio.sleep(0.05)

        requests = of_type(
            transport.to_client(active_client), "jester_choice_required")
        assert len(requests) == 1
        assert {p["client_id"] for p in requests[0]["players"]} == set(THREE_PLAYERS)
        assert {p["name"] for p in requests[0]["players"]} == set(THREE_PLAYERS.values())
        for client_id in THREE_PLAYERS:
            if client_id != active_client:
                assert of_type(
                    transport.to_client(client_id), "jester_choice_required") == []
        assert snapshot(game_state) == baseline  # no step happened
        assert bridge.engine.next_player_chooser is None

        # Invalid target: error to the chooser, still awaiting, no step.
        await bridge.handle_player_action(
            active_client, {"type": "jester_choose", "target_client_id": "ghost"})
        assert of_type(transport.to_client(active_client), "error")
        assert snapshot(game_state) == baseline
        assert not task.done()

        # Only the ACTING client may choose: c3's choice is rejected.
        await bridge.handle_player_action(
            "c3", {"type": "jester_choose", "target_client_id": "c2"})
        assert of_type(transport.to_client("c3"), "error")
        assert snapshot(game_state) == baseline
        assert not task.done()

        # Valid target (by client_id): chooser injected, the step proceeds,
        # and the finally-block clears the chooser afterwards.
        await bridge.handle_player_action(
            active_client, {"type": "jester_choose", "target_client_id": "c2"})
        info = await task
        assert info["result"] == "jester"
        assert game_state.get_active_player().name == "Bob"
        assert bridge.engine.next_player_chooser is None
        # 3 players: tavern 41 (36 numbers + 4 animals + 1 jester) + castle 12.
        assert total_cards(game_state) == 53

    asyncio.run(scenario())


def test_bridge_wrong_client_rejected(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state
    active_client = bridge.client_of_player[game_state.get_active_player()]
    other_client = next(cid for cid in TWO_PLAYERS if cid != active_client)
    baseline = snapshot(game_state)

    # Non-active client tries to play → error, no state change.
    info = asyncio.run(bridge.handle_player_action(
        other_client, {"type": "play", "card_indices": [0]}))
    assert info is None
    errors = of_type(transport.to_client(other_client), "error")
    assert errors and "turn" in errors[-1]["message"].lower()
    assert snapshot(game_state) == baseline
    assert bridge.engine.done is False
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS

    # Unregistered client id → rejected too.
    info = asyncio.run(bridge.handle_player_action(
        "ghost", {"type": "play", "card_indices": [0]}))
    assert info is None
    assert of_type(transport.to_client("ghost"), "error")
    assert snapshot(game_state) == baseline


def test_bridge_game_ended_loss(make_bridge, transport, total_cards):
    bridge = make_bridge(TWO_PLAYERS, seed=11)
    asyncio.run(bridge.initialize_game())
    game_state = bridge.engine.game_state

    # Drive deterministically to a loss: always play the highest single
    # card (hands dwindle fastest) and cover defenses greedily. Eventually
    # a player cannot cover the enemy attack → player_defeated.
    actions = 0
    while not bridge.engine.done and actions < 150:
        active = game_state.get_active_player()
        client = bridge.client_of_player[active]
        if bridge.engine.phase == "attack":
            if active.hand:
                index = max(range(len(active.hand)),
                            key=lambda i: active.hand[i].value)
                indices = [index]
            else:
                indices = []  # yield (the engine then kills/stuck-ends)
            asyncio.run(bridge.handle_player_action(
                client, {"type": "play", "card_indices": indices}))
        else:
            needed = game_state.current_enemy.effective_attack()
            asyncio.run(bridge.handle_player_action(
                client,
                {"type": "defend", "card_indices": cover_indices(active.hand, needed)}))
        actions += 1
        assert total_cards(game_state) == FULL_CAMPAIGN_CARDS

    assert bridge.engine.done is True
    assert bridge.game_over is True
    ended = of_type(transport.broadcasts, "game_ended")
    assert ended and ended[-1]["result"] == "player_defeated"

    # Terminal session: any further action is refused, state untouched.
    frozen = snapshot(game_state)
    client = next(iter(TWO_PLAYERS))
    info = asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": []}))
    assert info is None
    assert any(m["type"] == "error" and "not running" in m["message"].lower()
               for m in transport.to_client(client))
    assert snapshot(game_state) == frozen
