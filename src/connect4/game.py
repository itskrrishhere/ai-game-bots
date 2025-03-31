import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math

# ---------------------------
# Helper Functions for Connect4 Simulation
# ---------------------------
def get_valid_moves_sim(board):
    # A valid move is a column (0-6) whose top cell is empty.
    return [col for col in range(7) if board[0, col] == ' ']

def get_drop_row_sim(board, col):
    # Return the lowest empty row in the given column (None if column is full)
    for row in range(5, -1, -1):
        if board[row, col] == ' ':
            return row
    return None

def check_line_sim(board, row, col, d_row, d_col, player):
    cells = []
    for i in range(4):
        r = row + i * d_row
        c = col + i * d_col
        if r < 0 or r >= 6 or c < 0 or c >= 7:
            return []  # Out of bounds
        cells.append((r, c))
    if all(board[r, c] == player for r, c in cells):
        return cells
    return []

def check_winner_sim(board, player):
    for row in range(6):
        for col in range(7):
            if (check_line_sim(board, row, col, 1, 0, player) or
                    check_line_sim(board, row, col, 0, 1, player) or
                    check_line_sim(board, row, col, 1, 1, player) or
                    check_line_sim(board, row, col, 1, -1, player)):
                return True
    return False

def is_draw_sim(board):
    return np.all(board != ' ')

def evaluate_connect4(board):
    # Simple evaluation:
    # Return a high positive value if X wins, high negative if O wins, else 0.
    if check_winner_sim(board, 'X'):
        return 1000
    elif check_winner_sim(board, 'O'):
        return -1000
    else:
        return 0

# ---------------------------
# Minimax for Connect4 (with depth limit)
# ---------------------------
def minimax_connect4(board, depth, maximizing, use_alpha_beta, alpha, beta, depth_limit):
    # Terminal conditions: win for X or O, draw, or depth limit reached.
    if check_winner_sim(board, 'X') or check_winner_sim(board, 'O') or is_draw_sim(board) or depth == depth_limit:
        return evaluate_connect4(board)

    valid_moves = get_valid_moves_sim(board)
    if maximizing:
        best_score = -math.inf
        for col in valid_moves:
            row = get_drop_row_sim(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = 'X'
            score = minimax_connect4(new_board, depth + 1, False, use_alpha_beta, alpha, beta, depth_limit)
            best_score = max(best_score, score)
            if use_alpha_beta:
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = math.inf
        for col in valid_moves:
            row = get_drop_row_sim(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = 'O'
            score = minimax_connect4(new_board, depth + 1, True, use_alpha_beta, alpha, beta, depth_limit)
            best_score = min(best_score, score)
            if use_alpha_beta:
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score

def get_best_move_minimax_connect4(board, use_alpha_beta, depth_limit=4):
    valid_moves = get_valid_moves_sim(board)
    best_move = None
    best_score = -math.inf
    for col in valid_moves:
        row = get_drop_row_sim(board, col)
        if row is None:
            continue
        new_board = board.copy()
        new_board[row, col] = 'X'
        score = minimax_connect4(new_board, 0, False, use_alpha_beta, -math.inf, math.inf, depth_limit)
        if score > best_score:
            best_score = score
            best_move = col
    return best_move

# ---------------------------
# Q-Learning Stub for Connect4 (random move)
# ---------------------------
def get_best_move_qlearning_connect4(board):
    moves = get_valid_moves_sim(board)
    if moves:
        return random.choice(moves)
    return None

# ---------------------------
# Default Opponent for Connect4 (Rules-Based)
# ---------------------------
def default_opponent_move_connect4(board):
    valid_moves = get_valid_moves_sim(board)
    # Rule 1: If there is a winning move for O, take it.
    for col in valid_moves:
        row = get_drop_row_sim(board, col)
        if row is not None:
            temp_board = board.copy()
            temp_board[row, col] = 'O'
            if check_winner_sim(temp_board, 'O'):
                return col
    # Rule 2: Block winning move for X.
    for col in valid_moves:
        row = get_drop_row_sim(board, col)
        if row is not None:
            temp_board = board.copy()
            temp_board[row, col] = 'X'
            if check_winner_sim(temp_board, 'X'):
                return col
    # Rule 3: Otherwise, choose a random valid move.
    return random.choice(valid_moves) if valid_moves else None

# ---------------------------
# Connect4Game Simulation (Algorithm Agent vs Default Opponent)
# ---------------------------
class Connect4GameSim:
    def __init__(self, algorithm_choice):
        # algorithm_choice: 1 = minimax (no AB), 2 = minimax with AB, 3 = Q-learning (stub)
        self.algorithm_choice = algorithm_choice
        self.board = np.full((6, 7), ' ')
        self.game_over = False
        self.winning_cells = []  # Will store winning cells (if any)
        self.current_player = 'X'  # Algorithm agent is 'X'; default opponent is 'O'
        self.play_game()

    def play_game(self):
        while not self.game_over:
            if self.current_player == 'X':
                # Algorithm agent move
                move = self.algorithm_move()
                if move is None:
                    break
                row = get_drop_row_sim(self.board, move)
                self.board[row, move] = 'X'
                if check_winner_sim(self.board, 'X'):
                    self.game_over = True
                    self.winning_cells = self.get_winning_cells('X')
                elif is_draw_sim(self.board):
                    self.game_over = True
                else:
                    self.current_player = 'O'
            else:
                # Default opponent move
                move = default_opponent_move_connect4(self.board)
                if move is None:
                    break
                row = get_drop_row_sim(self.board, move)
                self.board[row, move] = 'O'
                if check_winner_sim(self.board, 'O'):
                    self.game_over = True
                    self.winning_cells = self.get_winning_cells('O')
                elif is_draw_sim(self.board):
                    self.game_over = True
                else:
                    self.current_player = 'X'
        self.draw_board()

    def algorithm_move(self):
        if self.algorithm_choice == 1:
            # Minimax without alpha-beta pruning
            return get_best_move_minimax_connect4(self.board, use_alpha_beta=False)
        elif self.algorithm_choice == 2:
            # Minimax with alpha-beta pruning
            return get_best_move_minimax_connect4(self.board, use_alpha_beta=True)
        elif self.algorithm_choice == 3:
            # Q-learning stub (random move)
            return get_best_move_qlearning_connect4(self.board)
        else:
            moves = get_valid_moves_sim(self.board)
            return random.choice(moves) if moves else None

    def get_winning_cells(self, player):
        # Retrieve winning cells by checking all positions.
        for row in range(6):
            for col in range(7):
                cells = (check_line_sim(self.board, row, col, 1, 0, player) or
                         check_line_sim(self.board, row, col, 0, 1, player) or
                         check_line_sim(self.board, row, col, 1, 1, player) or
                         check_line_sim(self.board, row, col, 1, -1, player))
                if cells:
                    return cells
        return []

    def draw_board(self):
        fig, ax = plt.subplots()
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor('blue')
        ax.grid(True, linestyle='', linewidth=2, color='black')

        for i in range(6):
            for j in range(7):
                color = 'white'
                if self.board[i, j] == 'X':
                    color = 'red'
                elif self.board[i, j] == 'O':
                    color = 'yellow'
                circle = plt.Circle((j, 5 - i), 0.4, fc=color, edgecolor='black')
                ax.add_patch(circle)
                if (i, j) in self.winning_cells:
                    highlight = plt.Circle((j, 5 - i), 0.45, fc='none', edgecolor='white', linewidth=4)
                    ax.add_patch(highlight)
        if self.game_over:
            if self.winning_cells:
                fig.suptitle(f"Winner: { 'X' if check_winner_sim(self.board, 'X') else 'O' }", fontsize=16)
            else:
                fig.suptitle("Draw!", fontsize=16)
        else:
            fig.suptitle(f"Turn: {self.current_player}", fontsize=16)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)
        plt.show()

# ---------------------------
# Main Program
# ---------------------------
def main():
    print("Choose algorithm for Algorithm Agent (X) vs Default Opponent (O):")
    print("1: Minimax without alpha-beta pruning")
    print("2: Minimax with alpha-beta pruning")
    print("3: Tabular Q-learning (stub)")
    choice = input("Enter your choice (1/2/3): ").strip()
    if choice not in ['1', '2', '3']:
        print("Invalid choice, defaulting to Q-learning stub (random).")
        choice = '3'
    game = Connect4GameSim(int(choice))

if __name__ == "__main__":
    main()
