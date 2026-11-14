# Regicide - Terminal & Web Edition

[![Python tests](https://github.com/fbec76/regicide/actions/workflows/python-tests.yml/badge.svg)](https://github.com/fbec76/regicide/actions/workflows/python-tests.yml)

```
 _______  _______  _______ _________ _______ _________ ______   _______ 
(  ____ )(  ____ \(  ____ \\__   __/(  ____ \\__   __/(  __  \ (  ____ \
| (    )|| (    \/| (    \/   ) (   | (    \/   ) (   | (  \  )| (    \/
| (____)|| (__    | |         | |   | |         | |   | |   ) || (__    
|     __)|  __)   | | ____    | |   | |         | |   | |   | ||  __)   
| (\ (   | (      | | \_  )   | |   | |         | |   | |   ) || (      
| ) \ \__| (____/\| (___) |___) (___| (____/\___) (___| (__/  )| (____/\
|/   \__/(_______/(_______)\_______/(_______/\_______/(______/ (_______/
```

Regicide is a cooperative card game where players work together to defeat a series of royal enemies (Jacks, Queens, and Kings) using a shared deck of standard playing cards. Players must use strategy, teamwork, and careful hand management to overcome the enemies' attacks and special abilities.

## Game Description

- **Objective:** Defeat all 12 royal enemies (Jacks, Queens, Kings) by playing cards to deal damage, heal, and use special powers.
- **Players:** 2–4 (cooperative; solo mode is not supported, matching the official rules).
- **Gameplay:**
  - Players take turns playing cards from their hand to attack the current enemy.
  - Each suit has a unique power (Hearts heal, Diamonds draw, Clubs double damage, Spades shield the enemy).
  - Special cards (Animals/Aces and Jesters) provide unique effects and combos.
  - Players must manage their hand size and cooperate to avoid defeat.
  - The game is lost if at least one player is unable to play or discard as required.

## Ways to Play

### CLI

```bash
cd src
python main.py
```

Terminal front end for 2–4 players. Note: `main.py` must be run from the `src/`
directory (it uses a top-level `from __init__ import ...` that resolves
relative to `src/`).

### Web

```bash
# From the repository root
python src/web/server.py
```

Starts one process serving both the WebSocket game server
(`ws://localhost:8765`) and a static HTTP sidecar (`http://localhost:8766`).
Open <http://localhost:8766> in one browser tab per player.

The browser client is a dark-fantasy table built with vanilla HTML/CSS/JS —
no frameworks, no build step. Features: lobby with game create/join (share the
session ID with other players), desktop table view, click-to-select card
play with a live server-computed play preview, castle progress strip, and a
turn log. The launcher works with both `websockets` 12.x and newer versions
(13+), so any Python 3.12+ setup with a reasonably current `websockets`
package will run it.

## Architecture

- **Language:** Python 3.12+
- **Core:** `src/game/` — consolidated turn engine shared by all front ends.
  The `Engine` exposes a gym-style action-dict interface
  (`Engine.reset` / `Engine.step`) built on `_begin_turn`, `_attempt_yield`,
  `_execute_play`, and `_execute_defense`. Game rules faithfully follow the
  official Regicide rules (see Resources).
- **CLI:** `src/cli/` — terminal display and input handling, driven by a run
  loop over the same core engine.
- **Web:** `src/web/` — WebSocket server (`app.py`), launcher (`server.py`),
  and a bridge (`websocket_cli_bridge.py`) that connects remote players to
  the engine. The bridge never mutates game state directly; every action goes
  through the validated `Engine.step` boundary. The browser client lives in
  `src/web/static/`.
- **Extensibility:** clear separation between game logic and user interfaces;
  the `Engine.step` action-dict contract is a ready-made environment interface
  (see Roadmap).

## Installation

```bash
# Clone the repository
git clone https://github.com/fbec76/regicide.git
cd regicide

python -m venv .venv
# unix
source .venv/bin/activate
# OR windows
.venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

Requires Python 3.12+.

## Running Tests

```bash
.venv/bin/python -m pytest tests -q
```

77 tests across three suites:

- **Unit** — engine, rules, game state, and player behavior (`tests/`).
- **E2E** — full-game suite (`tests/e2e/`): seeded games, 52-card deck
  conservation, wins, and a scripted loss.
- **Web** — bridge handler tests, wire integration with two real clients, and
  the server launcher (`tests/web/`).

> **Troubleshooting:** always run pytest via the project venv
> (`.venv/bin/python -m pytest`), not your system Python. A stale-shebang
> venv or wrong interpreter can cause confusing import failures.

## Project Layout

```
regicide/
├── src/
│   ├── main.py            # CLI entry point (run from src/)
│   ├── game/              # Core engine: card, deck, enemy, player,
│   │                      # rules, game_state, engine
│   ├── cli/               # Terminal display + input handling
│   ├── web/               # WebSocket server, launcher, engine bridge,
│   │   └── static/        # and static browser client (HTML/CSS/JS)
│   └── ml/                # Reserved for future ML work
├── tests/                 # Unit tests
│   ├── e2e/               # Full-game end-to-end tests
│   └── web/               # Bridge, wire-integration, launcher tests
└── requirements.txt
```

## Development

### Logging

The application supports different logging levels, including a DATA level for detailed game state information.
To enable this, create a `.env` file and set the logging level like so:

```bash
REGICIDE_LOG_LEVEL=DATA
```

## Known Limitations

- **Solo mode is unsupported** — 2–4 players only, per the official rules.
- **Jester next-player choice** in the CLI is answered at a terminal prompt;
  the web client handles it through a dedicated modal instead.
- **The game is intentionally hard.** Heuristic win rate on fresh 12-enemy
  campaigns is near zero (documented in the E2E test comments).

## Roadmap

- **ML / RL environment** — the `Engine.step` action-dict contract is the
  intended gym-style interface; the web bridge already serves as a reference
  consumer of that boundary.

## Resources

- [Official Regicide Rules PDF](https://www.regicidegame.com/site_files/33132/upload_files/RegicideRulesA4.pdf)
