from dotenv import load_dotenv

from cli.display import show_welcome

load_dotenv()

from __init__ import get_logger
from game.engine import Engine

logger = get_logger("regicide.main")


def ask_player_count():
    print("How many players will playing? (2-4)")
    while True:
        try:
            num_players = int(input("> ").strip())
            if num_players not in (2, 3, 4):
                print("2, 3, or 4 players only.")
                logger.warning(f"Invalid player count input: {num_players}")
                continue
            break
        except ValueError as e:
            print("Please enter a number (2-4).")
            logger.warning(f"Non-integer player count input: {e}")
    return num_players


def ask_player_names(num_players, max_hand_size):
    from game.player import Player

    players = []
    for i in range(num_players):
        name = input(f"Name for player {i + 1}: ").strip() or f"Player{i + 1}"
        players.append(Player(name=name, hand_limit=max_hand_size))
    return players


def main():
    hand_size_table = {2: 7, 3: 6, 4: 5}
    show_welcome()
    num_players = ask_player_count()
    max_hand_size = hand_size_table[num_players]
    players = ask_player_names(num_players, max_hand_size)
    engine = Engine(logger, get_logger)
    engine.run(players)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        logger = get_logger("regicide.main")
        logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        print("\033[91mA fatal error occurred. See regicide.log for details.\033[0m")
