# Regicide web client (static)

Vanilla HTML/CSS/JS browser client for the Regicide WebSocket game server.
No frameworks, no npm, no build step — ES modules (`type="module"`) only.

## Files

```
static/
├── index.html        three screens: lobby / game table / game over (+ jester modal)
├── css/style.css     dark fantasy table theme (CSS custom properties, gradients only)
└── js/
    ├── app.js        bootstrap: store + connection wiring, message routing table
    ├── connection.js reconnect-aware WebSocket wrapper (ws://localhost:8765)
    ├── state.js      central store + derived values (selection, log, turn flags)
    ├── ui.js         DOM renderers (all dynamic text via textContent)
    ├── actions.js    input handling (select/play/yield/defend/jester/leave)
    ├── preview.js    live play preview: request, tolerant parsing, rendering
    └── dom.js        shared text-only element factory
```

## Running

Everything runs from one process — the B1 backend serves this directory via
its static sidecar (WS on 8765, HTTP on 8766):

```bash
python src/web/server.py        # starts WS + static HTTP sidecar
```

Then open <http://localhost:8766/> in two (or more) browser tabs. No manual
`http.server` is needed anymore; the sidecar sets a strict CSP on every
response (`style-src 'self'` — no inline styles or handlers in the client).

**Custom ports:** unsupported by default. The client derives the WebSocket
port from the HTTP port the page is served on (sidecar convention: HTTP =
WS port + 1, so `8766 → 8765`); a sidecar started on a custom `ws_port` W
is served on W+1 and connects back to W. Serving the directory some other
way (e.g. a separate `http.server` on an unrelated port) or via `file://`
falls back to the hardcoded default `8765`.

## Protocol coverage

Client → server: `create_game`, `join_game`, `play {card_indices}` (empty
list = yield), `defend {card_indices}`, `jester_choose {target_client_id}`,
`preview {card_indices}`, `leave_game`.

Server → client: `welcome`, `error`, `game_created`, `player_joined`,
`player_left`, `game_started`, `state` (personalized), `action_required`,
`jester_choice_required`, `cards_played`, `player_yielded`,
`enemy_defeated`, `defense_resolved`, `action_rejected`, `game_ended`
(`win` / `loss` / `abandoned`). Unknown message types are silently ignored.

### Preview contract (live — backend handler implemented)

The client sends `{"type": "preview", "card_indices": [...]}` whenever the
selection changes during the owner's attack turn, and the server
(`websocket_cli_bridge.compute_preview`) answers with:

```json
{
  "type": "preview",
  "valid": true,
  "damage": 10,
  "double_damage": true,
  "powers": {
    "hearts": "fires",
    "diamonds": "blocked",
    "spades": "pending"
  },
  "defeats_enemy": false,
  "yield_attack": 7,
  "post_play_attack": 4,
  "pending_shield_value": 5
}
```

`post_play_attack` is the enemy's effective attack **after** this play
resolves (spade shields applied when they fire) — the client shows it as the
prominent "Enemy attack after this play: X" line. `pending_shield_value` is
`0` unless a spade is deferred to pending; that shield lands after a Jester,
and the client appends "shield deferred — after a Jester: attack reduced by
N more" when it is positive. `yield_attack` remains the yield-cost line
("If you yield: enemy strikes for N").

Per-suit three-state semantics: power **fires** / power **blocked but damage
still applies** / spade **shield deferred to `pending_spades`** (rendered as
`pending` — spade damage applies immediately; the shield lands retroactively
once a Jester negates immunity). Clubs have no power
entry — clubs only double damage (×2 marker via `double_damage`).
An illegal selection answers `{"valid": false, "reason": "…"}` and the
reason is shown inline in the preview panel (never in the turn log).
Transport errors during a pending preview are also shown inline.

### Shared state additions

Personalized `state` messages additionally carry `castle_defeated` (int —
enemies defeated so far) and `castle_total` (int — constant 12). The castle
progress strip renders `castle_total` slots, marking the first
`castle_defeated` as defeated (gold ✔). Older servers without these fields
fall back to the `castle_remaining` derivation.

## Security

**Every server-provided string is rendered via `textContent` /
`createTextNode`.** `innerHTML` with user- or server-derived data is
FORBIDDEN anywhere in `js/**` (grep-verifiable acceptance criterion, plan
B5). `index.html` contains the only static markup.
