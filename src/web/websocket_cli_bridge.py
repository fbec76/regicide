"""WebSocket bridge: a thin, non-mutating adapter over the game Engine.

The bridge owns transport, action mapping and state serialization only.
Every state change goes through ``Engine.reset``/``Engine.step`` — the
bridge performs ZERO writes to GameState/Player/Deck/Enemy fields
(grep-verifiable invariant: no ``<state>.<field> =`` anywhere here).

One-step-per-action (D4): each validated inbound action triggers exactly
one ``engine.step()``; there is no internal game loop and no pending-action
queue. ``step()`` is synchronous, so no ``await`` may sit between
index→Card resolution and the step call — the single exception is the
Jester peek-then-inject flow (D3), whose await is a pure choice request
during which the expected-actor guard blocks every state-changing action.

Stuck players: the bridge uses the REACTIVE path (no stuck pre-empt) —
``engine.step()`` runs the stuck check (``_begin_turn``) before action
parsing, so any action from a stuck player terminates the loss; a pre-empt
step would duplicate engine logic in the bridge for no gain.
"""
import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

from game.card import Card
from game.engine import Engine
from game.player import Player
from game.rules import can_play_jester, is_enemy_immune, is_valid_play, total_attack
from __init__ import get_logger

# Player count → initial hand size (mirrors main.py).
HAND_SIZE_TABLE = {2: 7, 3: 6, 4: 5}
# Generous window for the acting player to answer a Jester choice.
JESTER_CHOICE_TIMEOUT_SECONDS = 60.0
# Shown when a preview selection is not a legal play (mirrors rules.is_valid_play).
INVALID_PLAY_REASON = (
    "Not a valid play: play a single card, a same-value combo of 2-4 cards "
    "summing to at most 10, or an animal pair")
# Full castle size at reset (J/Q/K × 4 suits). One enemy is drawn into play
# at reset, so castle_defeated == CASTLE_TOTAL - castle_remaining
# - (1 if an enemy is in play).
CASTLE_TOTAL = 12


class WebSocketCLIBridge:
    """Adapter between the WebSocket server and the game Engine.

    Identity model: ``client_by_id`` maps each client_id to its Player and
    is built once at reset from the client-supplied names — no name-based
    client lookup exists anywhere (``jester_choose`` carries a
    ``target_client_id`` because the chooser picks a *player*, which is
    protocol, not lookup).
    """

    def __init__(
        self,
        session_id: str,
        client_players: Dict[str, str],
        broadcast_callback: Callable[[str, dict], Awaitable[None]],
        send_to_client_callback: Callable[[str, dict], Awaitable[None]],
        logger,
    ) -> None:
        self.session_id = session_id
        self.client_players = client_players  # {client_id: player_name}
        self.broadcast = broadcast_callback  # (session_id, message) → all clients
        self.send_to_client = send_to_client_callback  # (client_id, message) → one client
        self.logger = logger

        self.engine: Optional[Engine] = None
        self.client_by_id: Dict[str, Player] = {}
        self.client_of_player: Dict[Player, str] = {}
        self.game_over = False

        # Jester peek-then-inject state (D3): while an event is pending,
        # only the acting client's jester_choose is accepted.
        self._jester_event: Optional[asyncio.Event] = None
        self._jester_client_id: Optional[str] = None
        self._jester_choice: Optional[Player] = None

    # ------------------------------------------------------------------
    # Setup — the single game-setup path
    # ------------------------------------------------------------------

    async def initialize_game(self) -> None:
        """Build Players and hand ownership to ``Engine.reset``. The bridge
        never constructs TavernDeck/CastleDeck/GameState directly."""
        num_players = len(self.client_players)
        if num_players not in HAND_SIZE_TABLE:
            raise ValueError(
                f"Regicide supports 2-4 players (solo unsupported); got {num_players}")
        names = list(self.client_players.values())
        if len(set(names)) != len(names):
            # Uniqueness is enforced upstream in the lobby (A4); fail clean here.
            raise ValueError("Player names must be unique")
        hand_size = HAND_SIZE_TABLE[num_players]

        players = []
        for client_id, name in self.client_players.items():
            player = Player(name=name, hand_limit=hand_size)
            players.append(player)
            self.client_by_id[client_id] = player
            self.client_of_player[player] = client_id

        self.engine = Engine(self.logger, get_logger)
        self.engine.reset(players)
        self.game_over = False
        self.logger.info(
            f"Game initialized for session {self.session_id} ({num_players} players)")

        # Opening deal: personalized state to every client + first request.
        await self._send_personalized_states()
        await self._send_action_required()

    # ------------------------------------------------------------------
    # Inbound action routing (strict boundary validation)
    # ------------------------------------------------------------------

    async def handle_player_action(self, client_id: str, data: dict) -> Optional[dict]:
        """Handle one inbound message. Returns the engine ``info`` dict when a
        step executed, or None on any rejection (error already sent)."""
        if not isinstance(data, dict):
            await self._send_error(client_id, "Malformed action: expected a JSON object")
            return None
        msg_type = data.get("type")

        if self._jester_event is not None:
            if msg_type == "jester_choose":
                await self._handle_jester_choose(client_id, data)
            else:
                await self._send_error(
                    client_id, "A jester choice is required before any other action")
            return None

        if self.engine is None or self.game_over:
            await self._send_error(client_id, "The game is not running")
            return None

        if msg_type == "play":
            return await self._handle_play(client_id, data)
        if msg_type == "defend":
            return await self._handle_defend(client_id, data)
        if msg_type == "preview":
            await self._handle_preview(client_id, data)
            return None
        if msg_type == "jester_choose":
            await self._send_error(client_id, "No jester choice is currently required")
            return None
        await self._send_error(client_id, f"Unknown action type: {msg_type!r}")
        return None

    async def _handle_play(self, client_id: str, data: dict) -> Optional[dict]:
        error, player = self._guard_action(client_id, expected_phase="attack")
        if error:
            await self._send_error(client_id, error)
            return None
        indices = data.get("card_indices")
        error = self._validate_indices(indices, player.hand)
        if error:
            await self._send_error(client_id, error)
            return None
        cards = [player.hand[i] for i in indices]  # preserve client order

        action = {"type": "play", "cards": cards}
        if can_play_jester(cards):
            # D3 peek-then-inject: no state change until the choice arrives.
            chosen = await self._request_jester_choice(client_id, player)
            if chosen is None:
                return None  # timed out / aborted — error already sent
            self.engine.next_player_chooser = lambda players, _p=chosen: _p
            try:
                _, _, done, info = self.engine.step(action)
            finally:
                self.engine.next_player_chooser = None  # no stale chooser
        else:
            _, _, done, info = self.engine.step(action)
        await self._process_step_result(info, done, "play", cards, player)
        return info

    async def _handle_defend(self, client_id: str, data: dict) -> Optional[dict]:
        error, player = self._guard_action(client_id, expected_phase="defense")
        if error:
            await self._send_error(client_id, error)
            return None
        indices = data.get("card_indices")
        error = self._validate_indices(indices, player.hand)
        if error:
            await self._send_error(client_id, error)
            return None
        cards = [player.hand[i] for i in indices]
        _, _, done, info = self.engine.step({"type": "defend", "cards": cards})
        await self._process_step_result(info, done, "defend", cards, player)
        return info

    async def _handle_preview(self, client_id: str, data: dict) -> None:
        """Read-only play preview (D2). Same expected-client guard and index
        schema as play, but it never steps: index resolution and the
        computation are await-free, so no state change can interleave."""
        error, player = self._guard_action(client_id, expected_phase="attack")
        if error:
            await self._send_error(client_id, error)
            return
        indices = data.get("card_indices")
        error = self._validate_indices(indices, player.hand)
        if error:
            await self._send_error(client_id, error)
            return
        if not indices:
            # An empty play IS a yield (a state change) — never previewable.
            # The cost of yielding ships as ``yield_attack`` on every valid
            # preview instead.
            await self._send_error(
                client_id,
                "card_indices must not be empty (an empty play is a yield)")
            return
        cards = [player.hand[i] for i in indices]  # preserve client order
        preview = self.compute_preview(cards)
        await self.send_to_client(client_id, preview)

    def compute_preview(self, cards: List[Card]) -> dict:
        """Three-state preview computation (D2) — ZERO state mutation: only
        pure rules helpers plus read-only GameState inspection are used
        (grep-verifiable: no ``step()``, no field writes anywhere here)."""
        game_state = self.engine.game_state
        enemy = game_state.current_enemy
        jester_active = game_state.face_jester_played

        if not is_valid_play(cards):
            return {"type": "preview", "valid": False,
                    "reason": INVALID_PLAY_REASON}

        damage = total_attack(cards, jester_active, enemy=enemy)
        # Clubs doubling marker — mirrors total_attack's condition exactly
        # (there is no "clubs power": resolve_suit_power clubs is a pass).
        has_club = any(c.suit == "clubs" for c in cards if c.c_type != "jester")
        double_damage = has_club and not (
            enemy.suit == "clubs" and not jester_active)

        powers = {}
        for suit in ("hearts", "diamonds", "spades"):
            card_of_suit = next((c for c in cards if c.suit == suit), None)
            if card_of_suit is None:
                continue  # suit not in the selection → no power entry
            if not is_enemy_immune(enemy, card_of_suit, jester_active):
                powers[suit] = "fires"
            elif suit == "spades" and enemy.suit == "spades":
                # Deferred to pending_spades: the shield lands once a Jester
                # negates immunity (engine _execute_play collection gate).
                powers[suit] = "pending"
            else:
                # Immunity suppresses the power; the damage still applies.
                powers[suit] = "blocked"

        # Post-play attack (playtesting fix): the enemy's effective attack
        # immediately AFTER this play resolves. A firing spade lowers it
        # right away (floored at 0); a pending spade (spade vs spades enemy)
        # lands its shield only once a Jester negates the immunity — its
        # value is reported via pending_shield_value so the client can note
        # the deferred change. Blocked spades are structurally impossible
        # (spade immunity implies a spades enemy, which routes to pending),
        # so any spade not firing here is pending.
        firing_spade_values = [
            c.value for c in cards
            if c.suit == "spades" and powers.get("spades") == "fires"]
        pending_spade_values = [
            c.value for c in cards
            if c.suit == "spades" and powers.get("spades") == "pending"]
        post_play_attack = max(
            enemy.effective_attack() - sum(firing_spade_values), 0)
        pending_shield_value = sum(pending_spade_values)

        return {
            "type": "preview",
            "valid": True,
            "damage": damage,
            "double_damage": double_damage,
            "powers": powers,
            "defeats_enemy": enemy.current_damage + damage >= enemy.max_health,
            "yield_attack": enemy.effective_attack(),
            "post_play_attack": post_play_attack,
            "pending_shield_value": pending_shield_value,
        }

    def _guard_action(self, client_id: str, expected_phase: str):
        """Expected-actor guard: registered client, right phase, right turn.
        Returns (error_message, acting_player)."""
        player = self.client_by_id.get(client_id)
        if player is None:
            return "You are not a player in this game", None
        if self.engine.phase != expected_phase:
            return (f"Wrong phase: engine is in {self.engine.phase!r}, "
                    f"expected {expected_phase!r}"), None
        active = self.engine.game_state.get_active_player()
        if player is not active:
            return f"It is {active.name}'s turn", None
        return None, player

    @staticmethod
    def _validate_indices(indices, hand) -> Optional[str]:
        """Strict boundary validation for card_indices. Returns an error
        message or None. Empty list is legal (= yield for play actions)."""
        if not isinstance(indices, list):
            return "card_indices must be a list of integers"
        if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
            return "card_indices must be a list of integers"
        if len(set(indices)) != len(indices):
            return "card_indices must not contain duplicates"
        for i in indices:
            if i < 0 or i >= len(hand):
                return f"Card index {i} out of range (hand has {len(hand)} card(s))"
        return None

    # ------------------------------------------------------------------
    # Jester choice (D3 peek-then-inject)
    # ------------------------------------------------------------------

    async def _request_jester_choice(self, client_id: str, player: Player) -> Optional[Player]:
        """Ask the acting client to choose the next player. Returns the chosen
        Player, or None on timeout (play aborted, no state change)."""
        self._jester_choice = None
        self._jester_client_id = client_id
        self._jester_event = asyncio.Event()
        try:
            await self.send_to_client(client_id, {
                "type": "jester_choice_required",
                "player": player.name,
                "players": [
                    {"client_id": cid, "name": p.name}
                    for p in self.engine.game_state.players
                    for cid in (self.client_of_player.get(p),)
                    if cid is not None
                ],
                "message": "You played a Jester — choose which player takes the next turn",
            })
            try:
                await asyncio.wait_for(
                    self._jester_event.wait(), timeout=JESTER_CHOICE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Jester choice timed out for session {self.session_id}")
                await self._send_error(
                    client_id, "Jester choice timed out; the play was aborted")
                return None
        finally:
            self._jester_event = None
            self._jester_client_id = None
        return self._jester_choice

    async def _handle_jester_choose(self, client_id: str, data: dict) -> None:
        if client_id != self._jester_client_id:
            await self._send_error(
                client_id, "Only the acting player may choose the next player")
            return
        target_client_id = data.get("target_client_id")
        if not isinstance(target_client_id, str):
            await self._send_error(client_id, "target_client_id must be a string")
            return
        target = self.client_by_id.get(target_client_id)
        if target is None or target not in self.engine.game_state.players:
            await self._send_error(
                client_id, "Target is not a player in this session")
            return
        self._jester_choice = target
        self._jester_event.set()

    # ------------------------------------------------------------------
    # Step results → outbound messages
    # ------------------------------------------------------------------

    async def _process_step_result(self, info: dict, done: bool, action_type: str,
                                   cards, player: Player) -> None:
        if "error" in info:
            # Engine rejected the action: no mutation. Tell the actor, re-issue
            # the action request for the (unchanged) expected actor.
            client_id = self.client_of_player.get(player)
            await self.send_to_client(client_id, {
                "type": "action_rejected", "error": info["error"]})
            await self._send_action_required()
            return

        await self._broadcast_step_event(info, action_type, cards, player)
        if done:
            self.game_over = True
            await self.broadcast(self.session_id, {
                "type": "game_ended", "result": info.get("result")})
            await self._send_personalized_states()
            return

        await self._send_personalized_states()
        await self._send_action_required()

    async def _broadcast_step_event(self, info: dict, action_type: str,
                                    cards, player: Player) -> None:
        """Shared events derived ONLY from info['result'] — never recomputed."""
        result = info.get("result")
        if action_type == "play" and cards:
            await self.broadcast(self.session_id, {
                "type": "cards_played",
                "player": player.name,
                "cards": [c.display_name for c in cards],
                "jester": result == "jester",
            })
        if result in ("yielded_turn_advanced", "yielded_defense_required"):
            await self.broadcast(self.session_id, {
                "type": "player_yielded", "player": player.name})
        elif result == "enemy_defeated":
            await self.broadcast(self.session_id, {
                "type": "enemy_defeated", "player": player.name})
        elif result == "defense_success":
            await self.broadcast(self.session_id, {
                "type": "defense_resolved",
                "player": player.name,
                "discarded": [c.display_name for c in cards],
            })

    # ------------------------------------------------------------------
    # Outbound: personalized state + action requests
    # ------------------------------------------------------------------

    async def _send_personalized_states(self) -> None:
        for client_id in self.client_by_id:
            await self.send_to_client(client_id, self.serialize_state(client_id))

    async def _send_action_required(self) -> None:
        if self.engine is None or self.game_over:
            return
        game_state = self.engine.game_state
        active = game_state.get_active_player()
        client_id = self.client_of_player.get(active)
        if client_id is None:
            raise RuntimeError(
                f"Active player {active.name!r} has no registered client")
        message = {
            "type": "action_required",
            "phase": self.engine.phase,
            "player": active.name,
        }
        if self.engine.phase == "defense":
            needed = game_state.current_enemy.effective_attack()
            hand_sum = sum(c.value for c in active.hand)
            if hand_sum < needed:
                # Uncoverable defense (playtesting fix): the client gates
                # the defend button on hand sum, so an uncoverable player
                # never acts and the game hangs before the engine's
                # defense_validation could terminate the loss. Force the
                # empty defend — _execute_defense runs defense_validation
                # FIRST, which returns done=True with result
                # player_defeated; _process_step_result then broadcasts
                # game_ended. No action_required is sent in this case.
                self.logger.warning(
                    f"{active.name} cannot cover the attack of {needed} "
                    f"(hand sums to {hand_sum}); terminating as "
                    f"uncoverable-defense loss")
                _, _, done, info = self.engine.step(
                    {"type": "defend", "cards": []})
                await self._process_step_result(
                    info, done, "defend", [], active)
                return
            message["needed"] = needed
        else:
            message["can_yield"] = game_state.can_yield(active)
        await self.send_to_client(client_id, message)

    def serialize_state(self, for_client_id: str) -> dict:
        """Personalized, read-only snapshot for one client: full own hand,
        opponents as name + hand_size, shared board, active player's
        client_id revealed only to its owner."""
        game_state = self.engine.game_state
        active = game_state.get_active_player()

        you = None
        opponents = []
        for player in game_state.players:
            if player is self.client_by_id.get(for_client_id):
                you = self._serialize_own_player(player)
            else:
                opponents.append({"name": player.name, "hand_size": player.hand_size()})

        state = {
            "type": "state",
            "phase": self.engine.phase,
            "game_over": self.engine.done,
            "active_player": active.name,
            "enemy": self._serialize_enemy(game_state.current_enemy),
            "immunity": game_state.enemy_immunity,
            "face_jester_played": game_state.face_jester_played,
            "castle_remaining": len(game_state.castle_deck.cards),
            "castle_defeated": (
                CASTLE_TOTAL - len(game_state.castle_deck.cards)
                - (1 if game_state.current_enemy is not None else 0)),
            "castle_total": CASTLE_TOTAL,
            "tavern_count": len(game_state.tavern_deck.cards),
            "play_area": [c.display_name for c in game_state.cards_against_enemy],
            "you": you,
            "opponents": opponents,
        }
        if active is self.client_by_id.get(for_client_id):
            state["active_player_client_id"] = for_client_id
        return state

    @staticmethod
    def _serialize_enemy(enemy) -> Optional[dict]:
        if enemy is None:
            return None
        return {
            "name": enemy.display_name,
            "suit": enemy.suit,
            "rank": enemy.rank,
            "max_health": enemy.max_health,
            "current_damage": enemy.current_damage,
            "effective_attack": enemy.effective_attack(),
            "shield": enemy.shield,
        }

    @staticmethod
    def _serialize_own_player(player: Player) -> dict:
        return {
            "name": player.name,
            "hand_size": player.hand_size(),
            "hand": [
                {
                    "index": i,
                    "suit": c.suit,
                    "rank": c.rank,
                    "value": c.value,
                    "display_name": c.display_name,
                    "c_type": c.c_type,
                }
                for i, c in enumerate(player.hand)
            ],
        }

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    async def _send_error(self, client_id: str, message: str) -> None:
        await self.send_to_client(client_id, {"type": "error", "message": message})
