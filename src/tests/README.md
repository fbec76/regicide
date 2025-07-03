# Regicide Game Tests

This folder contains unit tests for the Regicide card game implementation.

## Test Structure

The tests are organized by component:

- `test_card.py`: Tests for the Card class
- `test_deck.py`: Tests for the Deck class and its subclasses (CastleDeck, TavernDeck, DiscardPile)
- `test_enemy.py`: Tests for the Enemy class
- `test_player.py`: Tests for the Player class
- `test_game_state.py`: Tests for the GameState class
- `test_rules.py`: Tests for the game rules functions

## Running the Tests

You can run all tests using the following command from the project root:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_card.py
```

To run a specific test case:

```bash
python -m unittest tests.test_card.TestCard
```

To run a specific test method:

```bash
python -m unittest tests.test_card.TestCard.test_is_animal
```

## Test Coverage

The tests cover the following functionality:

### Card Tests
- Card initialization
- Type checking (animal, jester, enemy)
- Card comparison
- String representation

### Deck Tests
- Deck initialization
- Shuffling
- Drawing cards
- Adding cards to top/bottom
- Checking deck state
- Castle deck creation
- Tavern deck creation with different player counts
- Discard pile functionality

### Enemy Tests
- Enemy initialization
- Damage application
- Shield management
- Defeat condition checking
- Attack calculation
- Enemy reset
- Immunity management
- String representation

### Player Tests
- Player initialization
- Card drawing and hand management
- Playing cards
- Discarding for damage
- Player status tracking
- String representation

### Game State Tests
- Game state initialization
- Turn management
- Enemy defeat handling
- Card play against enemies
- Shield and damage application
- Yield management
- Game win/loss condition checking
- String representation

### Rules Tests
- Card play validation (combos, animal pairs, jesters)
- Enemy immunity checking
- Suit power resolution
- Attack calculation
- Yield permission checking
- Enemy attack calculation
- Exact defeat checking