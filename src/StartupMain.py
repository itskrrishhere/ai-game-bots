import src.connect4.game
import src.tictactoe.game


def main():
    print("Choose game (1: tictactoe, 2: Connect4)")
    game_choice = int(input().strip())
    if game_choice == 1:
        src.tictactoe.game.startUp()
    else:
        src.connect4.game.startUp()


if __name__ == "__main__":
    main()
