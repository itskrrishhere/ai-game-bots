import ast
import hashlib
import json
import os
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import csv
import time
from collections import defaultdict

# ---------------------------
# Persistent Cache for Minimax Evaluations for Connect4
# ---------------------------
CACHE_FILENAME = "minimax_cache.json"
minimax_cache = {}

#Reference - https://github.com/KeithGalli/Connect4-Python/tree/master
def load_minimax_cache():
    global minimax_cache
    try:
        with open(CACHE_FILENAME, 'rt', encoding='utf-8') as f:
            minimax_cache = json.load(f)
            print(f"Cache loaded")
    except Exception as e:
        print(f"Cache load failed ({e}); starting with an empty cache.")
        minimax_cache = {}


def delete_minimax_cache():
    if os.path.exists(CACHE_FILENAME):
        os.remove(CACHE_FILENAME)


def board_to_key(board):
    return hashlib.md5(board.tobytes()).hexdigest()


# ---------------------------
# Helper Functions for Connect4
# ---------------------------
# Define the trainable configurations
ql_train_map = {3: 20000, 4: 60000, 5: 100000}


def get_algorithm_name(choice):
    algorithm_names = {
        1: "Minimax",
        2: "Minimax+AB",
        3: "Q-learning (20k episodes)",
        4: "Q-learning (60k episodes)",
        5: "Q-learning (100k episodes)",
        6: "Default"
    }
    return algorithm_names.get(choice, "Unknown Algorithm")


def get_valid_moves(board):
    # Valid moves are columns (0 to 6) that are not full.
    return [col for col in range(7) if board[0, col] == ' ']


def get_drop_row(board, col):
    for row in range(5, -1, -1):
        if board[row, col] == ' ':
            return row
    return None


def check_line(board, row, col, d_row, d_col, player):
    cells = []
    for i in range(4):
        r = row + i * d_row
        c = col + i * d_col
        if r < 0 or r >= 6 or c < 0 or c >= 7:
            return []
        cells.append((r, c))
    if all(board[r, c] == player for r, c in cells):
        return cells
    return []


def check_winner(board, player):
    for row in range(6):
        for col in range(7):
            if (check_line(board, row, col, 1, 0, player) or
                    check_line(board, row, col, 0, 1, player) or
                    check_line(board, row, col, 1, 1, player) or
                    check_line(board, row, col, 1, -1, player)):
                return True
    return False


def is_draw(board):
    return np.all(board != ' ')


def evaluate(board):
    if check_winner(board, 'X'):
        return 1000
    elif check_winner(board, 'O'):
        return -1000
    else:
        return 0


# ----------------------------------------------------------------
# Single-call Minimax that returns (score, best_move) in one go
# ----------------------------------------------------------------
def minimax_decision(board, player, use_alpha_beta=False, depth_limit=4):
    opponent = 'O' if player == 'X' else 'X'

    def minimax_recursive(rec_board, current_player, origin, depth, alpha, beta):
        # Create a JSON-serializable key using board hash and other parameters
        board_key = board_to_key(rec_board)
        key_tuple = (board_key, current_player, origin, depth, alpha, beta, use_alpha_beta)
        cache_key = json.dumps(key_tuple)
        if cache_key in minimax_cache:
            result = minimax_cache[cache_key]
            return result[0], result[1]

        # Terminal checks or depth cutoff
        if check_winner(rec_board, origin):
            val = 1000 - depth  # The earlier 'origin' wins, the better
            minimax_cache[cache_key] = (val, None)
            return val, None
        elif check_winner(rec_board, opponent):
            val = -1000 + depth
            minimax_cache[cache_key] = (val, None)
            return val, None
        elif is_draw(rec_board) or depth == depth_limit:
            val = evaluate(rec_board)
            minimax_cache[cache_key] = (val, None)
            return val, None

        valid_moves = get_valid_moves(rec_board)
        if not valid_moves:
            val = evaluate(rec_board)
            minimax_cache[cache_key] = (val, None)
            return val, None

        # Decide if we are maximizing or minimizing
        maximizing = (current_player == origin)
        best_move = random.choice(valid_moves)  # fallback if all else equal
        if maximizing:
            best_score = -math.inf
            for col in valid_moves:
                row = get_drop_row(rec_board, col)
                if row is None:
                    continue
                new_board = rec_board.copy()
                new_board[row, col] = current_player

                score, _ = minimax_recursive(new_board, opponent, origin, depth + 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = col
                if use_alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        else:
            best_score = math.inf
            for col in valid_moves:
                row = get_drop_row(rec_board, col)
                if row is None:
                    continue
                new_board = rec_board.copy()
                new_board[row, col] = current_player

                score, _ = minimax_recursive(new_board, origin, origin, depth + 1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = col
                if use_alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        minimax_cache[cache_key] = (best_score, best_move)
        return best_score, best_move

    # Single top-level call
    _, best_col = minimax_recursive(board, player, player, 0, -math.inf, math.inf)
    return best_col


# ---------------------------
# Q-Learning Agent for Connect4
# ---------------------------

class QLearningAgentConnect4:
    def __init__(self, player, alpha=0.5, gamma=0.999, epsilon=0.25, decay_factor=0.9999):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        self.prev_state = None
        self.prev_action = None
        self.state_history = []  # Track full state-action history

    def get_state_key(self, board):
        return tuple(map(tuple, board))

    def load_q_table(self, filename="qtable.json"):
        try:
            with open(filename, 'r') as f:
                serialized_qtable = json.load(f)

            # Convert string keys back to (state, action) tuples
            self.q_table.clear()
            for key_str, value in serialized_qtable.items():
                try:
                    key = ast.literal_eval(key_str)
                    if isinstance(key, tuple) and len(key) == 2:
                        self.q_table[key] = value
                except:
                    continue  # Skip invalid entries

            print(f"Q-table loaded from {filename}")
            return True

        except FileNotFoundError:
            print(f"Q-table file {filename} not found. Starting fresh.")
            return False
        except Exception as e:
            print(f"Error loading Q-table: {str(e)}")
            return False

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def save_qtable(self, filename="qtable.json"):
        try:
            # Convert (state, action) tuples to strings
            serializable_qtable = {
                str(key): value for key, value in self.q_table.items()
            }
            with open(filename, 'w') as f:
                json.dump(serializable_qtable, f, indent=4)  # indent for readability
            print(f"Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        valid_moves = get_valid_moves(np.array(next_state)) if next_state is not None else []
        next_max_q = max([self.get_q_value(next_state, a) for a in valid_moves] or [0.0])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def update_terminal_connect4(self, state, action, terminal_reward):
        self.update_q_value(state, action, terminal_reward, None)

    def get_intermediate_reward(self, board):
        player_count = np.sum(board == self.player)
        opponent_count = np.sum(board == self.opponent)
        return player_count - opponent_count

    def choose_action(self, board, training=True):
        state = self.get_state_key(board)
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            q_values = [(a, self.get_q_value(state, a)) for a in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1] if q_values else 0.0
            best_actions = [a for a, q in q_values if q == max_q]
            final_choice = random.choice(best_actions) if best_actions else None
            return final_choice

    def train(self, num_episodes=10000):
        win_count, lose_count = 0, 0
        for episode in range(num_episodes):
            board = np.full((6, 7), ' ')
            current_player = self.player  # Agent is 'X'
            state = self.get_state_key(board)
            done = False
            prev_state = None  # Track X's previous state
            prev_action = None  # Track X's previous action
            prev_reward = None  # Track intermediate reward for X

            while not done:
                if current_player == self.player:
                    # Agent's (X) turn
                    action = self.choose_action(board, training=True)
                    if action is None:
                        break

                    # Apply X's move
                    row = get_drop_row(board, action)
                    board[row, action] = self.player
                    next_state_x = self.get_state_key(board)

                    # Check if X wins or draws after their move
                    if check_winner(board, self.player):
                        reward = 50
                        win_count += 1
                        done = True
                        # Update Q-value for terminal state
                        self.update_q_value(state, action, reward, None)
                    elif is_draw(board):
                        reward = 10
                        done = True
                        self.update_q_value(state, action, reward, None)
                    else:
                        reward = 1  # Intermediate reward
                        # Store state/action/reward to update AFTER O's move
                        prev_state = state
                        prev_action = action
                        prev_reward = reward

                    # Switch to O's turn
                    current_player = self.opponent
                    state = next_state_x  # State after X's move
                else:
                    # Opponent's (O) turn
                    action = get_default_move(board, self.opponent)
                    if action is None:
                        break

                    # Apply O's move
                    row = get_drop_row(board, action)
                    board[row, action] = self.opponent
                    next_state_o = self.get_state_key(board)

                    # Check if O wins or draws after their move
                    if check_winner(board, self.opponent):
                        reward = -50
                        lose_count += 1
                        done = True
                        # Update Q-value for X's previous action (led to O's win)
                        if prev_state is not None:
                            self.update_q_value(prev_state, prev_action, reward, None)
                    elif is_draw(board):
                        reward = 10
                        done = True
                        if prev_state is not None:
                            self.update_q_value(prev_state, prev_action, reward, None)
                    else:
                        # Update Q-value using the state after O's move as next_state
                        if prev_state is not None:
                            self.update_q_value(prev_state, prev_action, prev_reward, next_state_o)

                    # Reset tracking variables
                    prev_state = None
                    prev_action = None
                    prev_reward = None

                    # Switch back to X's turn
                    current_player = self.player
                    state = next_state_o  # State after O's move

            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * self.decay_factor)
            if (episode + 1) % 1000 == 0:
                print(f"Q-learning ({self.player}) Episode {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.4f}")
                print(f"Win count: {win_count}, Loss count: {lose_count}")

        print(f"Final Win count: {win_count}, Loss count: {lose_count}")


# Global cache for Q-learning agents
q_agent_cache_connect4 = {}


def get_best_move_qlearning(board, player, episodes):
    global q_agent_cache_connect4
    key = (player, episodes)

    if key not in q_agent_cache_connect4:
        agent = QLearningAgentConnect4(player)
        filename = f"Qtrained_{player}_{episodes}"
        if not agent.load_q_table(filename):  # Try loading first
            print(f"Training new agent for {player} with {episodes} episodes...")
            agent.train(num_episodes=episodes)
            agent.save_qtable(filename)
        q_agent_cache_connect4[key] = agent

    return q_agent_cache_connect4[key].choose_action(board, training=False)


# ---------------------------
# Default Opponent for Connect4
# ---------------------------
def get_default_move(board, player):
    moves = get_valid_moves(board)
    # Check for winning move
    for move in moves:
        temp_board = board.copy()
        row = get_drop_row(temp_board, move)
        temp_board[row, move] = player
        if check_winner(temp_board, player):
            return move
    return random.choice(moves) if moves else None


# ---------------------------
# Connect4Game Simulation with optional Animation
# ---------------------------
class Connect4Game:
    def __init__(self, algorithm_x, algorithm_o, animate=True):
        self.algorithm_x = algorithm_x
        self.algorithm_o = algorithm_o
        self.board = np.full((6, 7), ' ')
        self.game_over = False
        self.winning_cells = []
        self.current_player = 'X'
        self.frames = []
        self.moves_info = []
        self.animate = animate
        self.result = None
        self.play_game()

    def get_algorithm_name(self, player):
        if player == 'X':
            return get_algorithm_name(self.algorithm_x)
        elif player == 'O':
            return get_algorithm_name(self.algorithm_o)
        return "Unknown"

    def play_game(self):
        self.frames.append(self.board.copy())
        self.moves_info.append("Game start")

        while not self.game_over:
            if self.current_player == 'X':
                move = self.get_move(self.algorithm_x, 'X')
            else:
                move = self.get_move(self.algorithm_o, 'O')

            if move is None:
                # No moves (should not happen unless board is full)
                break

            row = get_drop_row(self.board, move)
            self.board[row, move] = self.current_player
            self.moves_info.append(f"{self.current_player} moved to col {move}")
            self.frames.append(self.board.copy())

            if check_winner(self.board, self.current_player):
                self.game_over = True
                self.winning_cells = self.get_winning_cells(self.current_player)
                self.result = self.current_player
            elif is_draw(self.board):
                self.game_over = True
                self.result = 'Draw'
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'

        if self.winning_cells:
            algo_name = self.get_algorithm_name(self.result)
            self.moves_info.append(f"Winner: {algo_name} ({self.result})")
        else:
            x_algo = self.get_algorithm_name('X')
            o_algo = self.get_algorithm_name('O')
            self.moves_info.append(f"Draw between {x_algo} (X) and {o_algo} (O)")

        self.frames.append(self.board.copy())

        if self.animate:
            self.animate_game()

        return self.result

    def get_move(self, algorithm_choice, player):
        if algorithm_choice == 1:
            # Plain minimax (no alpha-beta), single call
            return minimax_decision(self.board, player, use_alpha_beta=False, depth_limit=4)
        elif algorithm_choice == 2:
            # Alpha-beta minimax, single call
            return minimax_decision(self.board, player, use_alpha_beta=True, depth_limit=4)
        elif algorithm_choice in (3, 4, 5):
            episodes = ql_train_map[algorithm_choice]
            return get_best_move_qlearning(self.board, player, episodes)
        elif algorithm_choice == 6:
            return get_default_move(self.board, player)
        else:
            # Fallback: random
            moves = get_valid_moves(self.board)
            return random.choice(moves) if moves else None

    def get_winning_cells(self, player):
        for row in range(6):
            for col in range(7):
                cells = (check_line(self.board, row, col, 1, 0, player) or
                         check_line(self.board, row, col, 0, 1, player) or
                         check_line(self.board, row, col, 1, 1, player) or
                         check_line(self.board, row, col, 1, -1, player))
                if cells:
                    return cells
        return []

    def animate_game(self):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            board_state = self.frames[frame]
            title = self.moves_info[frame] if frame < len(self.moves_info) else ""
            highlight = self.winning_cells if frame == len(self.frames) - 1 else []
            self.draw_board(ax, board_state, highlight, title)
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=750, repeat=False)
        plt.show()

    def draw_board(self, ax, board, highlight_cells=None, title=""):
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor('blue')
        for i in range(6):
            for j in range(7):
                color = 'white'
                if board[i, j] == 'X':
                    color = 'red'
                elif board[i, j] == 'O':
                    color = 'yellow'
                circle = plt.Circle((j, 5 - i), 0.4, fc=color, edgecolor='black')
                ax.add_patch(circle)
                if highlight_cells and (i, j) in highlight_cells:
                    highlight = plt.Circle((j, 5 - i), 0.45, fc='none', edgecolor='white', linewidth=4)
                    ax.add_patch(highlight)
        ax.set_title(title)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)


# ---------------------------
# Simulation for Connect4 (Including Time and Move Metrics)
# ---------------------------
def simulate_games(algorithm_x, algorithm_o, num_games=100):
    x_wins = 0
    o_wins = 0
    draws = 0
    total_time = 0.0
    total_moves = 0
    for _ in range(num_games):
        start_time = time.time()
        game = Connect4Game(algorithm_x, algorithm_o, animate=False)
        end_time = time.time()
        game_time = end_time - start_time
        total_time += game_time

        result = game.result
        if result == 'X':
            x_wins += 1
        elif result == 'O':
            o_wins += 1
        else:
            draws += 1

        moves_count = np.count_nonzero(game.board != ' ')
        total_moves += moves_count

    avg_time = total_time / num_games
    avg_moves = total_moves / num_games
    return x_wins, o_wins, draws, avg_time, avg_moves


def pre_train_qlearning_agents():
    def train_agent(player_key, episodes_key):
        key = (player_key, episodes_key)
        if key not in q_agent_cache_connect4:
            print(f"Pre-training Q-learning agent for Connect4 as {player_key}, {episodes_key} episodes...")
            agent = QLearningAgentConnect4(player_key)
            agent.train(num_episodes=episodes_key)
            q_agent_cache_connect4[key] = agent
            agent.save_qtable(f"Qtrained_{player_key}_{episodes_key}")

    # Create and start threads for each (player, episodes) pair
    for choice in (3, 4, 5):
        threads = []
        for player in ('X', 'O'):
            episodes = ql_train_map[choice]
            thread = threading.Thread(target=train_agent, args=(player, episodes))
            threads.append(thread)
            thread.start()
        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    print("Pre-training of Q-learning agents completed.")


# ---------------------------
# Main Program
# ---------------------------
def startUp():
    print("Choose simulation (1: Run a single animated game, 2: Run batch simulations)")
    sim_choice = int(input().strip())
    load_minimax_cache()  # Load persistent cache at the start
    if sim_choice == 1:
        print(
            "Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning (20k), 4: Q-learning (60k), "
            "5: Q-learning (100k), 6: Default):")
        choice_x = int(input().strip())
        print(
            "Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning (20k), 4: Q-learning (60k), "
            "5: Q-learning (100k), 6: Default):")
        choice_o = int(input().strip())
        for choice, player in [(choice_x, 'X'), (choice_o, 'O')]:
            if choice in (3, 4, 5):
                print("Choose (1: train , 2: load trained)")
                t_choice = int(input().strip())
                if t_choice == 1:
                    episodes = ql_train_map[choice]
                    key = (player, episodes)
                    if key not in q_agent_cache_connect4:
                        print(f"Training Q-learning agent for Connect4 as {player} with {episodes} episodes...")
                        agent = QLearningAgentConnect4(player)
                        agent.train(num_episodes=episodes)
                        q_agent_cache_connect4[key] = agent

        Connect4Game(choice_x, choice_o, animate=True)

    else:
        print("Choose (1: train , 2: load trained)")
        t_choice = int(input().strip())
        if t_choice == 1:
            pre_train_qlearning_agents()

        sim_num_games = 100  # Number of games per pairing

        # 1. Evaluation: Algorithms vs Default Opponent (Algorithm 6)
        eval_vs_default = {}
        for choice in range(1, 7):
            if choice == 6:  # Skip default vs default
                continue
            algo_name = get_algorithm_name(choice)
            print(f"Simulating {algo_name} (as X) vs Default Opponent (O)...")
            x_wins, o_wins, draws, avg_time, avg_moves = simulate_games(choice, 6, num_games=sim_num_games)
            eval_vs_default[algo_name] = (x_wins, o_wins, draws, avg_time, avg_moves)
        default_csv = "vs_default_results.csv"
        with open(default_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Algorithm", "Wins (%)", "Losses (%)", "Draws (%)", "Avg Game Time (s)", "Avg Moves"])
            for algo, (x_wins, o_wins, draws, avg_time, avg_moves) in eval_vs_default.items():
                total = x_wins + o_wins + draws
                win_rate = round((x_wins / total) * 100, 2) if total else 0
                loss_rate = round((o_wins / total) * 100, 2) if total else 0
                draw_rate = round((draws / total) * 100, 2) if total else 0
                writer.writerow([algo, win_rate, loss_rate, draw_rate, f"{avg_time:.4f}", f"{avg_moves:.2f}"])
        print(f"Vs Default evaluation results saved to {default_csv}")

        algos = list(eval_vs_default.keys())
        win_rates = [round((eval_vs_default[a][0] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]
        loss_rates = [round((eval_vs_default[a][1] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]
        draw_rates = [round((eval_vs_default[a][2] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]

        x_axis = np.arange(len(algos))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_axis - width, win_rates, width, label='Win Rate')
        ax.bar(x_axis, loss_rates, width, label='Loss Rate')
        ax.bar(x_axis + width, draw_rates, width, label='Draw Rate')
        ax.set_ylabel('Percentage')
        ax.set_title('Algorithms vs Default Opponent')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(algos)
        ax.legend()
        plt.tight_layout()
        plt.savefig("vs_default_graph.png", bbox_inches='tight')
        plt.close()
        print("Vs Default graph saved as vs_default_graph.png")

        # 2. Evaluation: Head-to-Head (Only for algorithms 1 to 5)
        eval_head2head = {}
        for i in range(1, 6):
            for j in range(i + 1, 6):
                algo_i = get_algorithm_name(i)
                algo_j = get_algorithm_name(j)
                print(f"Simulating {algo_i} (X) vs {algo_j} (O)...")
                x_wins, o_wins, draws, avg_time, avg_moves = simulate_games(i, j, num_games=sim_num_games)
                eval_head2head[(algo_i, algo_j)] = (x_wins, o_wins, draws, avg_time, avg_moves)
        head2head_csv = "head2head_results.csv"
        with open(head2head_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Pairing (X vs O)", "X Wins (%)", "O Wins (%)", "Draws (%)", "Avg Game Time (s)", "Avg Moves"])
            for (algo_i, algo_j), (x_wins, o_wins, draws, avg_time, avg_moves) in eval_head2head.items():
                total = x_wins + o_wins + draws
                x_win_rate = round((x_wins / total) * 100, 2) if total else 0
                o_win_rate = round((o_wins / total) * 100, 2) if total else 0
                draw_rate = round((draws / total) * 100, 2) if total else 0
                pairing = f"{algo_i} vs {algo_j}"
                writer.writerow([pairing, x_win_rate, o_win_rate, draw_rate, f"{avg_time:.4f}", f"{avg_moves:.2f}"])
        print(f"Head-to-Head evaluation results saved to {head2head_csv}")

        pairings = list(eval_head2head.keys())
        x_win_rates = []
        o_win_rates = []
        draw_rates = []
        for pairing in pairings:
            x_w, o_w, d, _, _ = eval_head2head[pairing]
            total = x_w + o_w + d
            x_win_rates.append(round((x_w / total) * 100, 2) if total else 0)
            o_win_rates.append(round((o_w / total) * 100, 2) if total else 0)
            draw_rates.append(round((d / total) * 100, 2) if total else 0)
        x_axis = np.arange(len(pairings))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_axis - width, x_win_rates, width, label='X Win Rate')
        ax.bar(x_axis, o_win_rates, width, label='O Win Rate')
        ax.bar(x_axis + width, draw_rates, width, label='Draw Rate')
        ax.set_ylabel('Percentage')
        ax.set_title('Head-to-Head Evaluation (Algorithms 1-5)')
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f"{p[0]} vs {p[1]}" for p in pairings], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig("head2head_graph.png", bbox_inches='tight')
        plt.close()
        print("Head-to-Head graph saved as head2head_graph.png")

        # 3. Overall Evaluation: Aggregate performance for each algorithm (1-6)
        overall_metrics = {alg: {"wins": 0, "games": 0, "time": 0, "moves": 0} for alg in range(1, 7)}
        # From vs Default (for algorithms 1-5)
        for choice in range(1, 6):
            algo_name = get_algorithm_name(choice)
            x_wins, o_wins, draws, avg_time, avg_moves = eval_vs_default[algo_name]
            total = x_wins + o_wins + draws
            overall_metrics[choice]["wins"] += x_wins
            overall_metrics[choice]["games"] += total
            overall_metrics[choice]["time"] += avg_time * total
            overall_metrics[choice]["moves"] += avg_moves * total
            # For default (as opponent), update from perspective of algorithm 6:
            overall_metrics[6]["wins"] += o_wins
            overall_metrics[6]["games"] += total
            overall_metrics[6]["time"] += avg_time * total
            overall_metrics[6]["moves"] += avg_moves * total
        # From head-to-head among algorithms 1-5
        for i in range(1, 6):
            for j in range(i + 1, 6):
                algo_i = get_algorithm_name(i)
                algo_j = get_algorithm_name(j)
                if (algo_i, algo_j) in eval_head2head:
                    x_wins, o_wins, draws, avg_time, avg_moves = eval_head2head[(algo_i, algo_j)]
                    total = x_wins + o_wins + draws
                    overall_metrics[i]["wins"] += x_wins
                    overall_metrics[i]["games"] += total
                    overall_metrics[i]["time"] += avg_time * total
                    overall_metrics[i]["moves"] += avg_moves * total
                    overall_metrics[j]["wins"] += o_wins
                    overall_metrics[j]["games"] += total
                    overall_metrics[j]["time"] += avg_time * total
                    overall_metrics[j]["moves"] += avg_moves * total

        overall_results = {}
        for alg in range(1, 7):
            games = overall_metrics[alg]["games"]
            if games > 0:
                win_rate = round((overall_metrics[alg]["wins"] / games) * 100, 2)
                avg_time_overall = overall_metrics[alg]["time"] / games
                avg_moves_overall = overall_metrics[alg]["moves"] / games
            else:
                win_rate = 0
                avg_time_overall = 0
                avg_moves_overall = 0
            overall_results[get_algorithm_name(alg)] = (win_rate, avg_time_overall, avg_moves_overall)

        overall_csv = "overall_results.csv"
        with open(overall_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Algorithm", "Overall Win Rate (%)", "Avg Game Time (s)", "Avg Moves"])
            for algo, (win_rate, avg_time_overall, avg_moves_overall) in overall_results.items():
                writer.writerow([algo, win_rate, f"{avg_time_overall:.4f}", f"{avg_moves_overall:.2f}"])
        print(f"Overall evaluation results saved to {overall_csv}")

        algos = list(overall_results.keys())
        overall_win_rates = [overall_results[a][0] for a in algos]
        overall_avg_times = [overall_results[a][1] for a in algos]
        overall_avg_moves = [overall_results[a][2] for a in algos]

        x = np.arange(len(algos))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, overall_win_rates, width, label='Win Rate (%)')
        ax.bar(x, overall_avg_times, width, label='Avg Game Time (s)')
        ax.bar(x + width, overall_avg_moves, width, label='Avg Moves')
        ax.set_title('Overall Evaluation of Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.legend()
        plt.tight_layout()
        plt.savefig("overall_graph.png", bbox_inches='tight')
        plt.close()
        print("Overall graph saved as overall_graph.png")

    delete_minimax_cache()


if __name__ == "__main__":
    startUp()
