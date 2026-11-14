"""Real-server integration tests (plan step A5 — Phase A gate).

Drives ``RegicideWebSocketServer`` with real ``websockets`` clients over the
wire on an EPHEMERAL port, using plain ``asyncio.run`` (no pytest-asyncio).

Documented approaches (required by plan step A5):

**Ephemeral port** — ``RegicideWebSocketServer`` takes ``host``/``port``
explicitly (app.py ctor); ``port=0`` makes ``websockets.serve`` bind an
ephemeral port and the real port is read from the returned server's socket
(``srv.sockets[0].getsockname()[1]``). No app.py edit, no monkeypatching.

**Deterministic seeding** — the server runs in the SAME pytest process (the
tests drive it from the same asyncio loop), and all engine randomness flows
through the global ``random`` module (see tests/e2e/conftest.py). Seeding
``random.seed(11)`` BEFORE ``create_game`` makes the deal deterministic.
There is no seed race: client_ids come from ``uuid4`` (os.urandom, not the
``random`` module), and nothing else consumes randomness between the seed
and ``Engine.reset``. Seed 11 mirrors the proven policy of
``tests/web/test_bridge.py::test_bridge_game_ended_loss`` (highest card +
greedy defense), which deterministically terminates in ``player_defeated``.

**Conservation == 52** — asserted via a SERVER-SIDE hook: the test reaches
the live ``WebSocketCLIBridge`` instance the app holds
(``server.active_games[session_id]``, same-process read-only access) and
runs the six-bucket helper on its ``game_state``. Rationale: the wire state
does not expose the discard-pile count, so client-received data alone cannot
reach == 52. Soundness: ``engine.step()`` is synchronous inside the
single-threaded event loop, so every client-received event is dispatched
strictly AFTER the step that produced it — checking on every event therefore
asserts conservation after every handled action.

**Jester flow over the wire (plan step A6).** A5 verified a transport
deadlock: ``client_handler`` processed each connection's messages
sequentially, so a jester play blocked the read loop inside
``WebSocketCLIBridge._request_jester_choice`` awaiting ``_jester_event`` and
the SAME client's ``jester_choose`` was never read (2s-timeout abort every
time; the choice cannot arrive via another connection either — jester_choose
is validated against the acting client's client_id). A6 fixes the transport
in ``app.py`` (per-message ``asyncio.create_task`` dispatch) and
``test_jester_flow_over_wire`` below pins the flow over the wire; it failed
pre-fix with the timeout abort and passes post-fix.

Every wait is bounded (5s per recv) so failures fail fast instead of hanging.
"""
import asyncio
import json
import random

import pytest
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from game.card import Card
from web.app import RegicideWebSocketServer

# The single origin app.py's cross-origin defense allows (A4 security).
ALLOWED_ORIGIN = "http://localhost:8766"
# 2 players: tavern 40 + castle 12 = 52 cards, 0 Jesters (mirrors conftest).
FULL_CAMPAIGN_CARDS = 52
RECV_TIMEOUT = 5.0
MAX_GAME_ACTIONS = 300


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

class Client:
    """Minimal websocket test client: records every received message."""

    def __init__(self, ws):
        self.ws = ws
        self.messages = []
        self.state = None  # latest personalized state

    async def send(self, message):
        await self.ws.send(json.dumps(message))

    async def recv(self, timeout=RECV_TIMEOUT):
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        message = json.loads(raw)
        self.messages.append(message)
        if message.get("type") == "state":
            self.state = message
        return message

    async def recv_until(self, message_type, timeout=RECV_TIMEOUT):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            assert remaining > 0, f"timed out waiting for {message_type!r}"
            message = await self.recv(timeout=remaining)
            if message.get("type") == message_type:
                return message

    def of_type(self, message_type):
        return [m for m in self.messages if m.get("type") == message_type]


async def start_ephemeral_server():
    """RegicideWebSocketServer ctor accepts host/port (no app.py edit):
    port=0 → ephemeral bind; resolve the real port from the listener."""
    server = RegicideWebSocketServer(host="127.0.0.1", port=0)
    srv = await server.start_server()
    port = srv.sockets[0].getsockname()[1]
    return server, srv, port


async def connect_client(port):
    ws = await websockets.connect(
        f"ws://127.0.0.1:{port}", origin=ALLOWED_ORIGIN, open_timeout=RECV_TIMEOUT)
    client = Client(ws)
    welcome = await client.recv()
    assert welcome["type"] == "welcome"
    return client


async def shutdown_server(srv, clients):
    for client in clients:
        await client.ws.close()
    srv.close()  # websockets 12.0 legacy: close() is sync, wait_closed async
    await srv.wait_closed()


def assert_conservation(server, session_id, total_cards):
    """Server-side conservation hook (documented above): six buckets == 52
    on the live bridge instance the app holds. Checked after every
    client-received event, i.e. after every handled action."""
    bridge = server.active_games.get(session_id)
    if bridge is None or bridge.engine is None:
        return
    assert total_cards(bridge.engine.game_state) == FULL_CAMPAIGN_CARDS


def assert_privacy(client):
    """No client ever receives another player's hand contents: opponents are
    counts-only ({name, hand_size}); full card detail only for the owner."""
    for state in client.of_type("state"):
        for opponent in state["opponents"]:
            assert set(opponent) == {"name", "hand_size"}
        assert "hand" in state["you"]


async def pump(client, queue, stop):
    """Background reader for the game-driving loop."""
    while not stop.is_set():
        try:
            message = await client.recv(timeout=0.1)
        except asyncio.TimeoutError:
            continue
        except ConnectionClosed:
            return
        queue.put_nowait((client, message))


async def respond_to_action_required(client, message):
    """Scripted policy — identical to the proven bridge loss policy in
    tests/web/test_bridge.py: highest single card on attack (first maximal
    index), empty play (= yield) when the hand is out, greedy highest-value
    cover on defense."""
    phase = message["phase"]
    hand = client.state["you"]["hand"]  # [{index, value, ...}], index-ascending
    if phase == "attack":
        if hand:
            indices = [max(hand, key=lambda entry: entry["value"])["index"]]
        else:
            indices = []  # yield
        await client.send({"type": "play", "card_indices": indices})
    else:
        needed = message["needed"]
        order = sorted(range(len(hand)),
                       key=lambda i: hand[i]["value"], reverse=True)
        picked, covered = [], 0
        for i in order:
            if covered >= needed:
                break
            picked.append(i)
            covered += hand[i]["value"]
        await client.send({"type": "defend", "card_indices": picked})


async def wait_for_event(queue, message_type, client=None, timeout=RECV_TIMEOUT):
    """Pop pumped events until one of message_type arrives (optionally from
    one specific client). Non-matching events are dropped — the game-driving
    loop only needs the handshake markers; everything relevant to play is
    handled by drive_to_loss."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        assert remaining > 0, f"timed out waiting for {message_type!r}"
        event_client, message = await asyncio.wait_for(
            queue.get(), timeout=remaining)
        if (message.get("type") == message_type
                and (client is None or event_client is client)):
            return event_client, message


async def drive_game(server, clients, queue, total_cards, handshake):
    """The SINGLE queue consumer from connection setup through terminal (the
    opening action_required arrives before game_started, so nothing may
    siphon events off the queue in between). Resolves game_created into
    ``handshake``, answers every action_required with the scripted policy,
    and returns once BOTH clients have received the terminal game_ended."""
    actions = 0
    while True:
        client, message = await asyncio.wait_for(queue.get(), timeout=RECV_TIMEOUT)
        assert_conservation(server, handshake.get("session_id"), total_cards)
        message_type = message.get("type")
        if message_type == "game_created":
            handshake["session_id"] = message["session_id"]
            handshake["created"].set()
            continue
        if message_type in ("welcome", "player_joined", "game_started"):
            continue
        if message_type == "game_ended":
            if all(c.of_type("game_ended") for c in clients):
                return
            continue
        if message_type == "action_required":
            actions += 1
            assert actions <= MAX_GAME_ACTIONS, (
                f"scripted game did not reach a terminal within "
                f"{MAX_GAME_ACTIONS} actions")
            await respond_to_action_required(client, message)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_game_two_clients_to_loss(total_cards):
    asyncio.run(_full_game_to_loss(total_cards))


async def _full_game_to_loss(total_cards):
    random.seed(11)  # deterministic deal; see module docstring
    server, srv, port = await start_ephemeral_server()
    clients = []
    queue = asyncio.Queue()
    stop = asyncio.Event()
    pumps = []
    try:
        alice = await connect_client(port)
        bob = await connect_client(port)
        clients = [alice, bob]
        # Pumps start BEFORE the handshake, and drive_game is the single
        # queue consumer: the bridge sends states and the opening
        # action_required BEFORE app.py broadcasts game_started, so nothing
        # may siphon events off the queue in between.
        pumps = [asyncio.ensure_future(pump(c, queue, stop)) for c in clients]
        handshake = {"session_id": None, "created": asyncio.Event()}
        # drive_game runs as a task from the start: it is the single queue
        # consumer and resolves the create handshake via ``handshake``.
        driver = asyncio.ensure_future(
            drive_game(server, clients, queue, total_cards, handshake))

        await alice.send({"type": "create_game", "num_players": 2,
                          "player_name": "Alice"})
        await asyncio.wait_for(handshake["created"].wait(), timeout=RECV_TIMEOUT)
        session_id = handshake["session_id"]

        await bob.send({"type": "join_game", "session_id": session_id,
                        "player_name": "Bob"})
        await asyncio.wait_for(driver, timeout=120)

        # Opening deal was personalized: 7-card own hand, counts-only
        # opponent (asserted from the first recorded state per client).
        for client, own_name in ((alice, "Alice"), (bob, "Bob")):
            first_state = client.of_type("state")[0]
            assert first_state["you"]["name"] == own_name
            assert first_state["you"]["hand_size"] == 7
            assert len(first_state["you"]["hand"]) == 7
            assert all(set(o) == {"name", "hand_size"}
                       for o in first_state["opponents"])

        # Loss terminal broadcast to BOTH clients.
        for client in clients:
            ended_msgs = client.of_type("game_ended")
            assert ended_msgs and ended_msgs[-1] == {
                "type": "game_ended", "result": "player_defeated"}

        # Server-side terminal state.
        bridge = server.active_games[session_id]
        assert bridge.engine.done is True
        assert bridge.game_over is True
        assert_conservation(server, session_id, total_cards)

        # Privacy across every state message either client ever received.
        for client in clients:
            assert client.of_type("state")
            assert_privacy(client)
    finally:
        stop.set()
        for pump_task in pumps:
            pump_task.cancel()
        if pumps:
            await asyncio.gather(*pumps, return_exceptions=True)
        await shutdown_server(srv, clients)


def test_lobby_flows_over_wire():
    asyncio.run(_lobby_flows())


async def _lobby_flows():
    server, srv, port = await start_ephemeral_server()
    clients = []

    async def new_client():
        client = await connect_client(port)
        clients.append(client)
        return client

    try:
        # Foreign origin rejected: cross-origin defense answers HTTP 403.
        with pytest.raises(InvalidStatusCode) as excinfo:
            await websockets.connect(
                f"ws://127.0.0.1:{port}", origin="http://evil.example",
                open_timeout=RECV_TIMEOUT)
        assert "403" in str(excinfo.value)

        # num_players=5 rejected at the boundary (no room created).
        stranger = await new_client()
        await stranger.send({"type": "create_game", "num_players": 5,
                             "player_name": "Host"})
        error = await stranger.recv_until("error")
        assert "num_players" in error["message"]
        assert not server.waiting_rooms

        # Valid create (num_players=3) + join x2 → game starts.
        host = await new_client()
        await host.send({"type": "create_game", "num_players": 3,
                         "player_name": "Host"})
        created = await host.recv_until("game_created")
        session_id = created["session_id"]
        assert created["players_needed"] == 3

        joiners = []
        for name in ("Second", "Third"):
            joiner = await new_client()
            await joiner.send({"type": "join_game", "session_id": session_id,
                               "player_name": name})
            joiners.append(joiner)
        await host.recv_until("game_started")
        for joiner in joiners:
            await joiner.recv_until("game_started")

        joined = host.of_type("player_joined")
        assert [(m["player_name"], m["players_count"], m["players_needed"])
                for m in joined] == [("Second", 2, 3), ("Third", 3, 3)]
        for client in [host, *joiners]:
            assert client.state["you"]["hand_size"] == 6  # 3-player hand size
            assert len(client.state["you"]["hand"]) == 6
            assert len(client.state["opponents"]) == 2
            assert_privacy(client)

        # Duplicate player name rejected; the rejected client can then join
        # under a fresh name (rejection must not consume the client).
        dup_host = await new_client()
        await dup_host.send({"type": "create_game", "num_players": 2,
                             "player_name": "Dup"})
        dup_created = await dup_host.recv_until("game_created")
        dup_joiner = await new_client()
        await dup_joiner.send({"type": "join_game",
                               "session_id": dup_created["session_id"],
                               "player_name": "Dup"})
        error = await dup_joiner.recv_until("error")
        assert "already taken" in error["message"]
        await dup_joiner.send({"type": "join_game",
                               "session_id": dup_created["session_id"],
                               "player_name": "Other"})
        await dup_host.recv_until("game_started")

        # leave_game during an in-flight game → game_ended {abandoned} for
        # everyone, and the session is removed.
        leave_host = await new_client()
        await leave_host.send({"type": "create_game", "num_players": 2,
                               "player_name": "Leaver"})
        leave_created = await leave_host.recv_until("game_created")
        leave_guest = await new_client()
        await leave_guest.send({"type": "join_game",
                                "session_id": leave_created["session_id"],
                                "player_name": "Guest"})
        await leave_host.recv_until("game_started")
        await leave_guest.send({"type": "leave_game"})
        for client in (leave_host, leave_guest):
            ended = await client.recv_until("game_ended")
            assert ended == {"type": "game_ended", "result": "abandoned"}
        assert leave_created["session_id"] not in server.active_games

        # Room cap: 8 concurrent waiting rooms + games; the 9th is refused.
        room_hosts = []
        for _ in range(12):
            if len(server.waiting_rooms) + len(server.active_games) >= 8:
                break
            room_host = await new_client()
            await room_host.send({"type": "create_game", "num_players": 2,
                                  "player_name": f"CapHost{len(room_hosts)}"})
            await room_host.recv_until("game_created")
            room_hosts.append(room_host)
        assert len(server.waiting_rooms) + len(server.active_games) == 8
        await stranger.send({"type": "create_game", "num_players": 2,
                             "player_name": "Overflow"})
        error = await stranger.recv_until("error")
        assert "full" in error["message"].lower()
    finally:
        await shutdown_server(srv, clients)


def test_jester_flow_over_wire(monkeypatch, total_cards):
    asyncio.run(_jester_flow_over_wire(monkeypatch, total_cards))


async def _jester_flow_over_wire(monkeypatch, total_cards):
    """Wire-level Jester flow (plan step A6): play a Jester, answer the
    choice request on the SAME connection, and assert the rotation happened.

    Pre-fix (sequential client_handler) this deadlocks: the jester play
    awaits ``_jester_event`` inside the connection's read loop, so the
    follow-up ``jester_choose`` is never read and the play aborts with the
    2s timeout error. The short monkeypatched window keeps that pre-fix
    failure fast instead of hanging for the 60s default."""
    monkeypatch.setattr(
        "web.websocket_cli_bridge.JESTER_CHOICE_TIMEOUT_SECONDS", 2.0)
    random.seed(0)  # deterministic deal; see module docstring
    server, srv, port = await start_ephemeral_server()
    clients = []
    try:
        alice = await connect_client(port)
        bob = await connect_client(port)
        clients = [alice, bob]
        await alice.send({"type": "create_game", "num_players": 2,
                          "player_name": "Alice"})
        created = await alice.recv_until("game_created")
        session_id = created["session_id"]
        await bob.send({"type": "join_game", "session_id": session_id,
                        "player_name": "Bob"})
        await alice.recv_until("game_started")
        await bob.recv_until("game_started")

        # Force a Jester into the acting player's hand via the live bridge
        # instance (a 2-player tavern holds no Jesters). The jester REPLACES
        # a hand card in place, so the 52-card total is unchanged.
        bridge = server.active_games[session_id]
        game_state = bridge.engine.game_state
        active = game_state.get_active_player()
        acting_client_id = bridge.client_of_player[active]
        client_by_id = {c.messages[0]["client_id"]: c for c in clients}
        acting_client = client_by_id[acting_client_id]
        active.hand[0] = Card("joker", "Jester", 0, "jester", "Jester")

        await acting_client.send({"type": "play", "card_indices": [0]})
        request = await acting_client.recv_until("jester_choice_required")
        assert [p["client_id"] for p in request["players"]] == \
            [c.messages[0]["client_id"] for c in clients]
        target_client_id = next(p["client_id"] for p in request["players"]
                                if p["client_id"] != acting_client_id)
        target_name = next(p["name"] for p in request["players"]
                           if p["client_id"] == target_client_id)

        await acting_client.send({"type": "jester_choose",
                                  "target_client_id": target_client_id})

        # The choice completes the play: post-step personalized state.
        state = await acting_client.recv_until("state")
        for client in clients:
            assert client.of_type("error") == [], (
                f"unexpected error(s): {client.of_type('error')}")
        played = acting_client.of_type("cards_played")
        assert played and played[-1]["jester"] is True
        # Rotation: the chosen target takes the next turn.
        assert state["active_player"] == target_name
        target_player = bridge.client_by_id[target_client_id]
        assert game_state.get_active_player() is target_player
        # The injected chooser never leaks past the step (bridge cleanup).
        assert bridge.engine.next_player_chooser is None
        assert bridge.engine.done is False
        assert_conservation(server, session_id, total_cards)
        assert_privacy(acting_client)
    finally:
        await shutdown_server(srv, clients)
