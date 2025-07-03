# Regicide

The card game Regicide playable via a CLI interface.

## Overview

Regicide is a cooperative card game where players work together to defeat the royal enemies (Jacks, Queens, and Kings) using number cards, animals (Aces), and jesters.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/regicide.git
cd regicide

# Install dependencies
pip install -r requirements.txt
```

## Running the Game

```bash
python src/main.py
```

## Testing

A comprehensive test suite is available in the `tests` directory. The tests cover all core components of the game:

- Card functionality
- Deck operations
- Player actions
- Enemy interactions
- Game state management
- Game rules

### Running Tests

You can run all tests using:

```bash
python tests/run_tests.py
```

Or using the unittest module:

```bash
python -m unittest discover tests
```

For more detailed information about the tests, see the [tests README](tests/README.md).
