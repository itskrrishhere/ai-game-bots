import math
import hashlib
import random

import numpy as np
from timeit import default_timer as timer

# Global variables
state_counter = {"minimax": 0, "alpha_beta": 0}
start_time = timer()
TIME_LIMIT = 10 * 60  # 10 minutes


def get_drop_row(board, col):
    for row in range(5, -1, -1):
        if board[row, col] == ' ':
            return row
    return None


def check_winner(board, player):
    for row in range(6):
        for col in range(7):
            if (check_line(board, row, col, 1, 0, player) or
                    check_line(board, row, col, 0, 1, player) or
                    check_line(board, row, col, 1, 1, player) or
                    check_line(board, row, col, 1, -1, player)):
                return True
    return False


def check_line(board, row, col, d_row, d_col, player):
    cells = []
    for i in range(4):
        r = row + i * d_row
        c = col + i * d_col
        if r < 0 or r >= 6 or c < 0 or c >= 7:
            return None
        cells.append((r, c))
    if all(board[r, c] == player for r, c in cells):
        return cells
    return None


def is_draw(board):
    return np.all(board != ' ')


def get_valid_moves(board):
    return [col for col in range(7) if board[0, col] == ' ']


def board_to_key(board, player):
    board_bytes = board.tobytes()
    return hashlib.md5(board_bytes + player.encode()).hexdigest()


def evaluate(board):
    if check_winner(board, 'X'):
        return 1000
    elif check_winner(board, 'O'):
        return -1000
    else:
        return 0


def alpha_beta_with_counter(board, current_player, origin, depth, alpha, beta, depth_limit):
    global state_counter, start_time
    if timer() - start_time > TIME_LIMIT:
        return 0, None

    state_counter["alpha_beta"] += 1
    opponent = 'O' if origin == 'X' else 'X'

    if check_winner(board, origin):
        return 1000 - depth, None
    if check_winner(board, opponent):
        return -1000 + depth, None
    if is_draw(board) or depth == depth_limit:
        return evaluate(board), None

    valid_moves = get_valid_moves(board)
    best_move = valid_moves[0] if valid_moves else None
    maximizing = (current_player == origin)

    if maximizing:
        value = -math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score, _ = alpha_beta_with_counter(new_board, 'O' if current_player == 'X' else 'X',
                                               origin, depth + 1, alpha, beta, depth_limit)
            if score > value:
                value = score
                best_move = col
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move
    else:
        value = math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score, _ = alpha_beta_with_counter(new_board, 'O' if current_player == 'X' else 'X',
                                               origin, depth + 1, alpha, beta, depth_limit)
            if score < value:
                value = score
                best_move = col
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move


def minimax_with_counter(board, current_player, origin, depth, depth_limit):
    global state_counter, start_time
    if timer() - start_time > TIME_LIMIT:
        return 0, None

    state_counter["minimax"] += 1
    opponent = 'O' if origin == 'X' else 'X'

    if check_winner(board, origin):
        return 1000 - depth, None
    if check_winner(board, opponent):
        return -1000 + depth, None
    if is_draw(board) or depth == depth_limit:
        return evaluate(board), None

    valid_moves = get_valid_moves(board)
    best_move = valid_moves[0] if valid_moves else random.choice(valid_moves)
    maximizing = (current_player == origin)

    if maximizing:
        best_score = -math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score, _ = minimax_with_counter(new_board, 'O' if current_player == 'X' else 'X',
                                            origin, depth + 1, depth_limit)
            if score > best_score:
                best_score = score
                best_move = col
        return best_score, best_move
    else:
        best_score = math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score, _ = minimax_with_counter(new_board, 'O' if current_player == 'X' else 'X',
                                            origin, depth + 1, depth_limit)
            if score < best_score:
                best_score = score
                best_move = col
        return best_score, best_move


def run_test(depth_limit=6):
    global start_time, state_counter

    board = np.full((6, 7), ' ')

    print("Running Minimax (no pruning) for X...")
    state_counter["minimax"] = 0
    start_time = timer()
    minimax_with_counter(board, 'X', 'X', 0, depth_limit)
    duration = timer() - start_time
    print(f"Minimax visited {state_counter['minimax']} nodes in {duration:.2f} seconds")

    print("\nRunning Minimax (no pruning) for O...")
    state_counter["minimax"] = 0
    start_time = timer()
    minimax_with_counter(board, 'O', 'O', 0, depth_limit)
    duration = timer() - start_time
    print(f"Minimax visited {state_counter['minimax']} nodes in {duration:.2f} seconds")

    print("\nRunning Minimax with Alpha-Beta pruning for X...")
    state_counter["alpha_beta"] = 0
    start_time = timer()
    alpha_beta_with_counter(board, 'X', 'X', 0, -math.inf, math.inf, depth_limit)
    duration = timer() - start_time
    print(f"Alpha-Beta visited {state_counter['alpha_beta']} nodes in {duration:.2f} seconds")

    print("\nRunning Minimax with Alpha-Beta pruning for O...")
    state_counter["alpha_beta"] = 0
    start_time = timer()
    alpha_beta_with_counter(board, 'O', 'O', 0, -math.inf, math.inf, depth_limit)
    duration = timer() - start_time
    print(f"Alpha-Beta visited {state_counter['alpha_beta']} nodes in {duration:.2f} seconds")


if __name__ == "__main__":
    run_test()
