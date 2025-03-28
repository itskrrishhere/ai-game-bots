import numpy as np
import matplotlib.pyplot as plt
import random

# Reference: https://en.wikipedia.org/wiki/Tic-tac-toe
class TicTacToeGame:
    def __init__(self):
        self.board = np.full((3, 3), ' ')
        self.current_player = 'X'  # Human is X; default opponent is O
        self.game_over = False
        self.winning_cells = []  # Stores winning positions
        self.fig, self.ax = plt.subplots()
        self.draw_board()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def draw_board(self):
        self.ax.clear()
        self.ax.set_xticks([0.5, 1.5, 2.5], minor=True)
        self.ax.set_yticks([0.5, 1.5, 2.5], minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Draw each cell with its value. Winning cells are highlighted.
        for i in range(3):
            for j in range(3):
                color = 'black'
                if (i, j) in self.winning_cells:
                    color = 'green'
                self.ax.text(j, 2 - i, self.board[i, j], fontsize=40, ha='center', va='center', color=color)

        # Title displays whose turn it is or the final outcome.
        if self.game_over:
            if self.winning_cells:
                self.fig.suptitle(f"Player {self.current_player} Wins!", fontsize=16)
            else:
                self.fig.suptitle("It's a Draw!", fontsize=16)
        else:
            self.fig.suptitle(f"Player {self.current_player}'s Turn", fontsize=16)

        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(-0.5, 2.5)
        plt.draw()

    def on_click(self, event):
        # Human move only if game is not over and coordinates are valid.
        if self.game_over or event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(2 - event.ydata))

        # Validate move
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == ' ':
            self.board[row, col] = 'X'  # Human is 'X'
            winner, cells = self.check_winner('X')
            if winner:
                self.game_over = True
                self.winning_cells = cells
            elif np.all(self.board != ' '):  # Check draw
                self.game_over = True
                self.winning_cells = []
            self.draw_board()

            # If game is still not over, let default opponent (O) make a move.
            if not self.game_over:
                self.default_opponent_move()
                # After opponent move, check for win/draw.
                winner, cells = self.check_winner('O')
                if winner:
                    self.game_over = True
                    self.winning_cells = cells
                elif np.all(self.board != ' '):
                    self.game_over = True
                    self.winning_cells = []
                else:
                    # Switch turn back to human.
                    self.current_player = 'X'
                self.draw_board()

    def default_opponent_move(self):
        """Default opponent's move based on the rules."""
        # Opponent is 'O'
        valid_moves = [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == ' ']

        # Rule 1: If a winning move is available, take it.
        for move in valid_moves:
            i, j = move
            self.board[i, j] = 'O'
            win, _ = self.check_winner('O')
            if win:
                self.current_player = 'O'
                return
            self.board[i, j] = ' '  # Revert move

        # Rule 2: Block opponent's (X) winning move.
        for move in valid_moves:
            i, j = move
            self.board[i, j] = 'X'
            win, _ = self.check_winner('X')
            if win:
                self.board[i, j] = 'O'  # Block move
                self.current_player = 'O'
                return
            self.board[i, j] = ' '  # Revert move

        # Rule 3: No immediate win or block; pick a random move.
        move = random.choice(valid_moves)
        i, j = move
        self.board[i, j] = 'O'
        self.current_player = 'O'

    def check_winner(self, player):
        # Check rows and columns
        for i in range(3):
            if all(self.board[i, j] == player for j in range(3)):
                return True, [(i, j) for j in range(3)]
            if all(self.board[j, i] == player for j in range(3)):
                return True, [(j, i) for j in range(3)]
        # Check main diagonal
        if all(self.board[i, i] == player for i in range(3)):
            return True, [(i, i) for i in range(3)]
        # Check anti-diagonal
        if all(self.board[i, 2 - i] == player for i in range(3)):
            return True, [(i, 2 - i) for i in range(3)]
        return False, []

# Run the game
TicTacToeGame()
