import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy
import math

# -----------------------------
# Helper functions for Tic Tac Toe
# -----------------------------

def get_valid_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i, j] == ' ':
                moves.append((i, j))
    return moves

def check_winner(board, player):
    # Check rows and columns
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True, [(i, j) for j in range(3)]
        if all(board[j, i] == player for j in range(3)):
            return True, [(j, i) for j in range(3)]
    # Check main diagonal
    if all(board[i, i] == player for i in range(3)):
        return True, [(i, i) for i in range(3)]
    # Check anti-diagonal
    if all(board[i, 2 - i] == player for i in range(3)):
        return True, [(i, 2 - i) for i in range(3)]
    return False, []

def is_draw(board):
    return np.all(board != ' ')

def evaluate(board):
    # Evaluation from the perspective of 'X'
    winX, _ = check_winner(board, 'X')
    winO, _ = check_winner(board, 'O')
    if winX:
        return 1
    elif winO:
        return -1
    else:
        return 0

# -----------------------------
# Minimax Algorithms
# -----------------------------

def minimax(board, depth, maximizing, use_alpha_beta=False, alpha=-math.inf, beta=math.inf):
    terminal, _ = check_winner(board, 'X')
    if terminal or check_winner(board, 'O')[0] or is_draw(board):
        return evaluate(board)

    moves = get_valid_moves(board)

    if maximizing:
        best_score = -math.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = 'X'
            score = minimax(new_board, depth + 1, False, use_alpha_beta, alpha, beta)
            best_score = max(best_score, score)
            if use_alpha_beta:
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = math.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = 'O'
            score = minimax(new_board, depth + 1, True, use_alpha_beta, alpha, beta)
            best_score = min(best_score, score)
            if use_alpha_beta:
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score

def get_best_move_minimax(board, use_alpha_beta=False):
    moves = get_valid_moves(board)
    best_move = None
    if board is None or len(moves) == 0:
        return best_move

    # Algorithm agent is 'X', maximizing player
    best_score = -math.inf
    for move in moves:
        new_board = board.copy()
        new_board[move[0], move[1]] = 'X'
        score = minimax(new_board, 0, False, use_alpha_beta)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# -----------------------------
# Q-Learning stub (for demonstration)
# -----------------------------
def get_best_move_qlearning(board):
    # In a full Q-learning implementation, one would use a learned Q-table.
    # Here we simply return a random valid move as a placeholder.
    moves = get_valid_moves(board)
    if moves:
        return random.choice(moves)
    return None

# -----------------------------
# Default Opponent (rules-based)
# -----------------------------
def default_opponent_move(board):
    # Opponent is 'O'
    moves = get_valid_moves(board)
    # Rule 1: Winning move for O.
    for move in moves:
        temp_board = board.copy()
        temp_board[move[0], move[1]] = 'O'
        if check_winner(temp_board, 'O')[0]:
            return move
    # Rule 2: Block winning move for X.
    for move in moves:
        temp_board = board.copy()
        temp_board[move[0], move[1]] = 'X'
        if check_winner(temp_board, 'X')[0]:
            return move
    # Rule 3: Otherwise, choose a random move.
    return random.choice(moves)

# -----------------------------
# TicTacToeGame Simulation (Algorithm vs Default Opponent) with Animation
# -----------------------------
class TicTacToeGame:
    def __init__(self, algorithm_choice):
        # algorithm_choice: 1 = minimax (no AB), 2 = minimax with AB, 3 = Q-learning (stub)
        self.algorithm_choice = algorithm_choice
        self.board = np.full((3, 3), ' ')
        self.game_over = False
        self.winning_cells = []  # Stores winning positions for final state
        self.current_player = 'X'  # Algorithm agent always plays as X; opponent is O.
        # List to store board states (deep copies) for animation
        self.frames = []
        # Also store the player who moved in each frame for annotation purposes
        self.moves_info = []
        self.play_game()

    def play_game(self):
        # Store the initial board state
        self.frames.append(self.board.copy())
        self.moves_info.append("Game start")

        while not self.game_over:
            if self.current_player == 'X':
                # Algorithm agent move
                move = self.algorithm_move()
                if move is None:
                    break  # No move available
                self.board[move[0], move[1]] = 'X'
                self.moves_info.append("X moved to " + str(move))
                self.frames.append(self.board.copy())
                win, cells = check_winner(self.board, 'X')
                if win:
                    self.game_over = True
                    self.winning_cells = cells
                elif is_draw(self.board):
                    self.game_over = True
                else:
                    self.current_player = 'O'
            else:
                # Default opponent move
                move = default_opponent_move(self.board)
                if move is None:
                    break
                self.board[move[0], move[1]] = 'O'
                self.moves_info.append("O moved to " + str(move))
                self.frames.append(self.board.copy())
                win, cells = check_winner(self.board, 'O')
                if win:
                    self.game_over = True
                    self.winning_cells = cells
                elif is_draw(self.board):
                    self.game_over = True
                else:
                    self.current_player = 'X'
        # Final frame annotation for win/draw
        if self.winning_cells:
            winner = 'X' if check_winner(self.board, 'X')[0] else 'O'
            self.moves_info.append(f"Winner: {winner}")
        else:
            self.moves_info.append("Draw!")
        # After game over, animate the play
        self.animate_game()

    def algorithm_move(self):
        if self.algorithm_choice == 1:
            # Use Minimax without alpha-beta pruning
            return get_best_move_minimax(self.board, use_alpha_beta=False)
        elif self.algorithm_choice == 2:
            # Use Minimax with alpha-beta pruning
            return get_best_move_minimax(self.board, use_alpha_beta=True)
        elif self.algorithm_choice == 3:
            # Use Q-Learning (stub)
            return get_best_move_qlearning(self.board)
        else:
            # Fallback to random move if invalid choice
            moves = get_valid_moves(self.board)
            return random.choice(moves) if moves else None

    def draw_board(self, ax, board, highlight_cells=None, title=""):
        ax.clear()
        # Set up grid
        ax.set_xticks([0.5, 1.5, 2.5], minor=True)
        ax.set_yticks([0.5, 1.5, 2.5], minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        # Draw X and O
        for i in range(3):
            for j in range(3):
                color = 'black'
                if highlight_cells and (i, j) in highlight_cells:
                    color = 'green'
                ax.text(j, 2 - i, board[i, j], fontsize=40, ha='center', va='center', color=color)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(title, fontsize=16)

    def animate_game(self):
        fig, ax = plt.subplots()

        # Map algorithm_choice to algorithm names
        algorithm_names = {
            1: "Minimax without alpha-beta pruning",
            2: "Minimax with alpha-beta pruning",
            3: "Tabular Q-learning (stub)"
        }

        def update(frame):
            board_state = self.frames[frame]
            # Highlight winning cells only on the final frame
            highlight = self.winning_cells if (frame == len(self.frames) - 1 and self.winning_cells) else None
            title = self.moves_info[frame] if frame < len(self.moves_info) else ""

            # For the final frame, update the figure's suptitle to show the game result
            if frame == len(self.frames) - 1:
                if self.winning_cells:
                    # Determine the winner based on the board state
                    if check_winner(board_state, 'X')[0]:
                        winner_name = algorithm_names.get(self.algorithm_choice, "Algorithm")
                    else:
                        winner_name = "Default Opponent"
                    fig.suptitle(f"Winner: {winner_name}", fontsize=16)
                else:
                    fig.suptitle("Draw!", fontsize=16)
            else:
                # Clear suptitle for intermediate frames
                fig.suptitle("")
            self.draw_board(ax, board_state, highlight_cells=highlight, title=title)
            return ax

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=1000, repeat=False)
        plt.show()


# -----------------------------
# Main Program
# -----------------------------
def main():
    print("Choose algorithm to play as X (Algorithm Agent) vs Default Opponent (O):")
    print("1: Minimax without alpha-beta pruning")
    print("2: Minimax with alpha-beta pruning")
    print("3: Tabular Q-learning (stub)")
    choice = input("Enter your choice (1/2/3): ").strip()
    if choice not in ['1', '2', '3']:
        print("Invalid choice, defaulting to random (Q-learning stub).")
        choice = '3'
    TicTacToeGame(int(choice))

if __name__ == "__main__":
    main()
