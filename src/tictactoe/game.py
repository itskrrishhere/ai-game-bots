import numpy as np
import matplotlib.pyplot as plt

# reference - https://en.wikipedia.org/wiki/Tic-tac-toe
class TicTacToeGame:
    def __init__(self):
        self.board = np.full((3, 3), ' ')
        self.current_player = 'X'
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

        for i in range(3):
            for j in range(3):
                color = 'black'
                if (i, j) in self.winning_cells:
                    color = 'green'  # Highlight winning cells
                self.ax.text(j, 2 - i, self.board[i, j], fontsize=40, ha='center', va='center', color=color)

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
        if self.game_over or event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(2 - event.ydata))

        if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == ' ':
            self.board[row, col] = self.current_player
            winner, cells = self.check_winner(self.current_player)

            if winner:
                self.game_over = True
                self.winning_cells = cells
            elif np.all(self.board != ' '):  # Check for a draw if the board is full
                self.game_over = True  # Draw case
                self.winning_cells = []  # No winning cells in case of draw

            if not self.game_over:
                self.current_player = 'O' if self.current_player == 'X' else 'X'

            self.draw_board()

    def check_winner(self, player):
        for i in range(3):
            # Row win
            if all(self.board[i, :] == player):
                return True, [(i, j) for j in range(3)]
            # Column win
            if all( self.board[:, i] == player):
                return True, [(j, i) for j in range(3)]
        # diagonal (top-left to bottom-right) win
        if all(np.diag(self.board) == player):
            return True, [(i, i) for i in range(3)]
        # diagonal (top-right to bottom-left) win
        if all(np.diag(np.fliplr(self.board)) == player):
            return True, [(i, 2 - i) for i in range(3)]

        return False, []

# Run the game
TicTacToeGame()
