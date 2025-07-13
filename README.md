# Regicide - Terminal Edition

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

- **Objective:** Defeat all royal enemies (Jacks, Queens, Kings) by playing cards to deal damage, heal, and use special powers.
- **Gameplay:**
  - Players take turns playing cards from their hand to attack the current enemy.
  - Each suit has a unique power (Hearts heal, Diamonds draw, Clubs double damage, Spades shield the enemy).
  - Special cards (Animals/Aces and Jesters) provide unique effects and combos.
  - Players must manage their hand size and cooperate to avoid defeat.
  - The game is lost if at least one players are unable to play or discard as required.

## Technical Overview

- **Language:** Python 3
- **Interface:** Command-line (CLI)
- **Structure:**
  - Modular codebase with separate modules for cards, decks, players, enemies, rules, and game state.
  - Uses OOP principles for game entities (Card, Player, Enemy, Deck, etc.).
  - CLI display and input handling for interactive gameplay.
  - Game logic closely follows the official Regicide rules.
- **Extensibility:**
  - Easily modifiable for new rules, card types, or UI improvements.
  - Clear separation between game logic and user interface.

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

## Running the Game

```bash
python src/main.py
```

## Running Tests

This project uses [pytest](https://docs.pytest.org/) for testing. To run all tests, make sure you have pytest installed:

```bash
pytest tests
```

This will automatically discover and run all test files in the `tests/` directory.

## Development

### Logging

The application supports different logging levels, including a DATA level for detailed game state information. 
To enable this, create and `.env` file and set the logging level like so:

```bash
REGICIDE_LOG_LEVEL=DATA
```
## Resources

- [Official Regicide Rules PDF](https://www.regicidegame.com/site_files/33132/upload_files/RegicideRulesA4.pdf)
