"""Lost-end hardening regressions (user playtest report: "the no-defence-
possible lost end game does not trigger (it also says 'defense against 0')").

Two empirical findings are pinned here:

1. **defense_needed in the shared state** — the client must never depend on
   action-message ordering for the defense requirement. A render between the
   `state` message and the `action_required` message (or a stale action after
   the auto-loss, which never sends one) previously fell back to
   "Discard cards worth at least 0". The state snapshot now carries
   ``defense_needed`` during the defense phase, and the client prefers it.

2. **Stuck-player loss pre-empt** — a player with an empty hand who cannot
   yield (both other-yield flags set) is RULED OUT of cards by the official
   rules, but the engine's stuck check only runs when an action arrives and
   that client has no enabled action. The bridge now terminates the loss at
   the point it would have sent the impossible action_required.

Bridge-level tests over a real Engine (mirrors test_playtesting_fixes.py).
"""
import asyncio

from test_bridge import FULL_CAMPAIGN_CARDS, of_type
from test_preview import TWO_PLAYERS


class WireTransport:
    """Broadcast-faithful transport: broadcasts are appended to EVERY
    client's stream at the broadcast point, exactly like app.py's
    broadcast_to_session — so per-client message ORDER is assertable."""

    def __init__(self):
        self.sent = {}
        self.broadcasts = []

    async def send_to_client(self, client_id, message):
        self.sent.setdefault(client_id, []).append(message)

    async def broadcast(self, session_id, message):
        self.broadcasts.append(message)
        for client_id in self.sent:
            self.sent[client_id].append(message)

    def to_client(self, client_id):
        return self.sent.get(client_id, [])

    def clear(self):
        self.sent = {}
        self.broadcasts = []


def _scan(make_bridge, predicate, max_seeds=4000):
    """First seed whose initialized bridge satisfies predicate."""
    for seed in range(max_seeds):
        bridge = make_bridge(TWO_PLAYERS, seed=seed)
        asyncio.run(bridge.initialize_game())
        if predicate(bridge):
            return bridge, seed
    raise AssertionError(f"seed scan exhausted ({max_seeds} seeds)")


def _drain_hand(bridge, player, keep):
    """Conservation-safe hand shaping: removed cards go to the discard pile."""
    gs = bridge.engine.game_state
    drained = [c for c in player.hand if c not in keep]
    for c in drained:
        gs.discard_pile.add_to_top(c)
    player.hand = list(keep)


# ---------------------------------------------------------------------------
# 1 — defense_needed in the shared state snapshot
# ---------------------------------------------------------------------------

def test_state_defense_needed_field(make_bridge, transport):
    def pred(bridge):
        gs = bridge.engine.game_state
        hand = gs.get_active_player().hand
        enemy = gs.current_enemy
        # a single non-spade number card whose play neither defeats the
        # enemy nor shields it away (attack stays > 0 → defense phase)
        card = next((c for c in hand
                     if c.c_type == "number" and c.suit != "spades"
                     and c.value < enemy.attack), None)
        return card is not None and enemy.attack > 0

    bridge, seed = _scan(make_bridge, pred)
    gs = bridge.engine.game_state
    active = gs.get_active_player()
    client = bridge.client_of_player[active]
    play_index = next(
        i for i, c in enumerate(active.hand)
        if c.c_type == "number" and c.suit != "spades" and c.value < gs.current_enemy.attack)

    transport.clear()
    asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": [play_index]}))

    assert bridge.engine.phase == "defense"
    states = of_type(transport.to_client(client), "state")
    assert states, "expected a state message after the play"

    # Attack-phase states (everything before the last) carry NO field.
    for s in states[:-1]:
        assert "defense_needed" not in s
    # The defense-phase state carries the true requirement — not 0.
    defense_state = states[-1]
    assert defense_state["phase"] == "defense"
    needed = gs.current_enemy.effective_attack()
    assert needed > 0
    assert defense_state["defense_needed"] == needed


# ---------------------------------------------------------------------------
# 2 — stuck player (empty hand, cannot yield) terminates the loss
# ---------------------------------------------------------------------------

def test_stuck_player_auto_loss(make_bridge, transport, total_cards):
    def pred(bridge):
        gs = bridge.engine.game_state
        enemy = gs.current_enemy
        hand = gs.get_active_player().hand
        # P1's whole hand must cover the enemy attack after yielding
        return enemy.attack > 0 and sum(c.value for c in hand) >= enemy.attack

    bridge, seed = _scan(make_bridge, pred)
    gs = bridge.engine.game_state
    p1 = gs.get_active_player()
    p1_client = bridge.client_of_player[p1]
    p2 = next(p for p in gs.players if p is not p1)
    p2_client = bridge.client_of_player[p2]

    # P2's hand emptied (conservation-safe): the coming turn is stuck if
    # P1 yields now (P1's yield flag forbids P2's yield).
    _drain_hand(bridge, p2, keep=[])

    # P1 yields → defense_required (attack > 0) → P1 defends with the hand.
    asyncio.run(bridge.handle_player_action(
        p1_client, {"type": "play", "card_indices": []}))
    defense = of_type(transport.to_client(p1_client), "action_required")[-1]
    assert defense["phase"] == "defense"
    needed = defense["needed"]

    hand_values = sorted(c.value for c in p1.hand)
    chosen, covered = [], 0
    for i, c in enumerate(p1.hand):
        if covered < needed:
            chosen.append(i)
            covered += c.value
    assert covered >= needed
    transport.clear()
    asyncio.run(bridge.handle_player_action(
        p1_client, {"type": "defend", "card_indices": chosen}))

    # P2's turn: stuck (empty hand, P1's yield forbids yielding). The bridge
    # must pre-empt the loss instead of asking for an impossible action.
    assert gs.player_is_stuck(p2)
    assert bridge.engine.done is True
    assert bridge.game_over is True
    ended = of_type(transport.broadcasts, "game_ended")
    assert ended and ended[-1]["result"] == "player_defeated"

    # NO action_required was sent for the impossible turn — to anyone.
    for cid in bridge.client_by_id:
        assert of_type(transport.to_client(cid), "action_required") == []

    # Both clients received the terminal state (post-loss snapshot).
    for cid in bridge.client_by_id:
        states = of_type(transport.to_client(cid), "state")
        assert states and states[-1]["game_over"] is True

    # Conservation intact and the session is terminal.
    assert total_cards(gs) == FULL_CAMPAIGN_CARDS
    result = asyncio.run(bridge.handle_player_action(
        p1_client, {"type": "play", "card_indices": [0]}))
    assert result is None
    errors = of_type(transport.to_client(p1_client), "error")
    assert errors and errors[-1]["message"] == "The game is not running"


# ---------------------------------------------------------------------------
# 3 — uncoverable-defense loss: message order per client
# ---------------------------------------------------------------------------

def test_uncoverable_loss_message_order(make_bridge, total_cards):
    def pred(bridge):
        gs = bridge.engine.game_state
        enemy = gs.current_enemy
        hand = gs.get_active_player().hand
        play = next((c for c in hand if c.c_type == "number"
                     and c.suit in ("hearts", "clubs")), None)
        return play is not None and enemy.attack > 0

    bridge, seed = _scan(make_bridge, pred)
    transport = WireTransport()
    bridge.broadcast = transport.broadcast
    bridge.send_to_client = transport.send_to_client
    gs = bridge.engine.game_state
    active = gs.get_active_player()
    client = bridge.client_of_player[active]
    play = next(c for c in active.hand if c.c_type == "number"
                and c.suit in ("hearts", "clubs"))
    kept = min((c for c in active.hand if c is not play and c.c_type == "number"),
               key=lambda c: c.value)
    assert kept.value < gs.current_enemy.attack
    _drain_hand(bridge, active, keep=[play, kept])
    play_index = active.hand.index(play)

    asyncio.run(bridge.handle_player_action(
        client, {"type": "play", "card_indices": [play_index]}))

    assert bridge.game_over is True
    for cid in bridge.client_by_id:
        stream = transport.to_client(cid)
        ended_positions = [i for i, m in enumerate(stream)
                           if m["type"] == "game_ended"]
        assert ended_positions, f"{cid} missed game_ended"
        # The terminal broadcast precedes the post-loss snapshot for every
        # client, and no action_required is ever sent for the lost defense.
        post_loss_state = [i for i, m in enumerate(stream)
                           if m["type"] == "state" and m.get("game_over") is True]
        assert post_loss_state
        assert ended_positions[0] < post_loss_state[0]
        assert of_type(stream, "action_required") == []

    # The post-loss state carries the (frozen) defense requirement, not 0:
    # a client rendering between messages can never show "at least 0".
    for cid in bridge.client_by_id:
        stream = transport.to_client(cid)
        post_loss_state = next(m for m in stream
                               if m["type"] == "state" and m.get("game_over") is True)
        assert post_loss_state["phase"] == "defense"
        assert post_loss_state["defense_needed"] > 0

    assert total_cards(gs) == FULL_CAMPAIGN_CARDS
