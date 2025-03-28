import numpy as np
import matplotlib.pyplot as plt

#Reference - https://github.com/KeithGalli/Connect4-Python/tree/master
class Connect4Game:
    def __init__(self):
        self.board = np.full((6, 7), ' ')
        self.current_player = 'X'
        self.game_over = False
        self.winning_cells = []  # Store winning positions
        self.fig, self.ax = plt.subplots()
        self.draw_board()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def draw_board(self):
        self.ax.clear()
        self.ax.set_xticks(range(7))
        self.ax.set_yticks(range(6))
        self.ax.set_xticklabels([])  # Remove x-tick labels
        self.ax.set_yticklabels([])  # Remove y-tick labels
        self.ax.set_facecolor('blue')
        self.ax.grid(True, linestyle='', linewidth=2, color='black')

        # Draw the game board
        for i in range(6):
            for j in range(7):
                color = 'white'
                if self.board[i, j] == 'X':
                    color = 'red'
                elif self.board[i, j] == 'O':
                    color = 'yellow'
                circle = plt.Circle((j, 5 - i), 0.4, fc=color, edgecolor='black')
                self.ax.add_patch(circle)

                if (i, j) in self.winning_cells:
                    highlight = plt.Circle((j, 5 - i), 0.45, fc='none', edgecolor='white', linewidth=4)
                    self.ax.add_patch(highlight)  # Highlight winning cells

        # Display winner or draw message
        if self.game_over:
            if self.winning_cells:
                self.fig.suptitle(f"Player {self.current_player} Wins!", fontsize=16)
            else:
                self.fig.suptitle("It's a Draw!", fontsize=16)
        else:
            self.fig.suptitle(f"Player {self.current_player}'s Turn", fontsize=16)

        self.ax.set_xlim(-0.5, 6.5)
        self.ax.set_ylim(-0.5, 5.5)
        plt.draw()

    def on_click(self, event):
        if self.game_over or event.xdata is None:
            return

        col = int(round(event.xdata))
        if col not in self.get_valid_moves():
            return

        # Drop the disc in the correct row
        for row in range(5, -1, -1):
            if self.board[row, col] == ' ':
                self.board[row, col] = self.current_player
                break

        # Check for winner
        winner, cells = self.check_winner(self.current_player)
        if winner:
            self.game_over = True
            self.winning_cells = cells
        elif np.all(self.board != ' '):  # Check for draw if board is full
            self.game_over = True  # It's a draw
            self.winning_cells = []  # No winning cells for a draw

        if not self.game_over:
            self.current_player = 'O' if self.current_player == 'X' else 'X'

        self.draw_board()

    def get_valid_moves(self):
        return [col for col in range(7) if self.board[0, col] == ' ']

    def check_winner(self, player):
        for row in range(6):
            for col in range(7):
                if (cells := self.check_line(row, col, 1, 0, player)) or \
                        (cells := self.check_line(row, col, 0, 1, player)) or \
                        (cells := self.check_line(row, col, 1, 1, player)) or \
                        (cells := self.check_line(row, col, 1, -1, player)):
                    return True, cells
        return False, []

    def check_line(self, row, col, d_row, d_col, player):
        cells = []
        for i in range(4):
            r = row + i * d_row
            c = col + i * d_col
            # Check bounds: rows must be 0 to 5 and cols 0 to 6
            if r < 0 or r >= 6 or c < 0 or c >= 7:
                return []  # Out of bounds, no valid winning line here
            cells.append((r, c))
        if all(self.board[r, c] == player for r, c in cells):
            return cells
        return []

# Run the game
Connect4Game()
