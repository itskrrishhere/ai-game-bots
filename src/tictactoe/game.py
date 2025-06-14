import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from collections import defaultdict

# Global cache for minimax evaluations
minimax_cache = {}


# reference - https://realpython.com/tic-tac-toe-python/
# -----------------------------
# Helper functions for Tic Tac Toe
# -----------------------------
def get_algorithm_name(choice):
    algorithm_names = {
        1: "Minimax",
        2: "Minimax+AB",
        3: "Q-learning (10k episodes)",
        4: "Q-learning (20k episodes)",
        5: "Q-learning (30k episodes)",
        6: "Default"
    }
    return algorithm_names.get(choice, "Unknown Algorithm")


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


# -----------------------------
# Minimax Algorithms with Caching/Memoization
# -----------------------------
def minimax(board, current_player, origin, use_alpha_beta=False, alpha=-math.inf, beta=math.inf):
    board_key = tuple(map(tuple, board))
    cache_key = (board_key, current_player, origin, use_alpha_beta)
    if cache_key in minimax_cache:
        return minimax_cache[cache_key]

    # Terminal condition: if board is finished, evaluate from origin's perspective.
    win_origin, _ = check_winner(board, origin)
    opponent = 'O' if origin == 'X' else 'X'
    win_opp, _ = check_winner(board, opponent)
    if win_origin or win_opp or is_draw(board):
        if win_origin:
            return 1
        elif win_opp:
            return -1
        else:
            return 0

    moves = get_valid_moves(board)

    if current_player == origin:
        best_score = -math.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = current_player
            next_player = opponent if current_player == origin else origin
            score = minimax(new_board, next_player, origin, use_alpha_beta, alpha, beta)
            best_score = max(best_score, score)
            if use_alpha_beta:
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
    else:
        best_score = math.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = current_player
            next_player = origin if current_player == opponent else opponent
            score = minimax(new_board, next_player, origin, use_alpha_beta, alpha, beta)
            best_score = min(best_score, score)
            if use_alpha_beta:
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

    minimax_cache[cache_key] = best_score
    return best_score


def get_best_move_minimax(board, player, use_alpha_beta=False):
    moves = get_valid_moves(board)
    best_move = None
    if not moves:
        return None

    best_score = -math.inf
    for move in moves:
        new_board = board.copy()
        new_board[move[0], move[1]] = player
        opponent = 'O' if player == 'X' else 'X'
        score = minimax(new_board, opponent, player, use_alpha_beta, -math.inf, math.inf)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# -----------------------------
# Q-Learning
# -----------------------------
class QLearningAgent:
    def __init__(self, player, alpha=0.5, gamma=0.99, epsilon=0.1, decay_factor=0.9999):
        self.player = player  # 'X' or 'O'
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_factor = decay_factor

    def get_state_key(self, board):
        return tuple(map(tuple, board))

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        valid_moves = get_valid_moves(np.array(next_state)) if next_state is not None else []
        next_max_q = max([self.get_q_value(next_state, a) for a in valid_moves] or [0.0])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

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
            return random.choice(best_actions) if best_actions else None

    def blocks_opponent(self, board, action):
        opponent = 'O' if self.player == 'X' else 'X'
        for move in get_valid_moves(board):
            test_board = board.copy()
            test_board[move[0], move[1]] = opponent
            if check_winner(test_board, opponent)[0]:
                test_board = board.copy()
                test_board[action[0], action[1]] = self.player
                test_board[move[0], move[1]] = opponent
                if not check_winner(test_board, opponent)[0]:
                    return True
        return False

    def train(self, num_episodes=10000):
        opponent = 'O' if self.player == 'X' else 'X'
        win_count, lose_count = 0, 0
        for episode in range(num_episodes):
            board = np.full((3, 3), ' ')
            current_player = 'X'
            prev_state = None
            prev_action = None
            done = False

            while not done:
                if current_player == self.player:
                    state = self.get_state_key(board)
                    action = self.choose_action(board, training=True)
                    if action is None:
                        break
                    new_board = board.copy()
                    new_board[action[0], action[1]] = self.player

                    win, _ = check_winner(new_board, self.player)
                    draw = is_draw(new_board)
                    reward = 0
                    if win:
                        reward = 20
                        win_count += 1
                        done = True
                    elif draw:
                        reward = 0
                        done = True
                    elif self.blocks_opponent(board, action):
                        reward = 0.25

                    if prev_state is not None and prev_action is not None:
                        self.update_q_value(prev_state, prev_action, reward, self.get_state_key(new_board))
                    prev_state = state
                    prev_action = action
                    board = new_board
                    current_player = opponent

                    if done:
                        self.update_q_value(state, action, reward, None)
                else:
                    # Use minimax for the opponent instead of the default move.
                    # move = get_best_move_minimax(board, current_player, use_alpha_beta=False)
                    # if move is None:
                    #     done = True
                    #     break
                    # new_board = board.copy()
                    # new_board[move[0], move[1]] = opponent
                    move = get_default_move(board, current_player)
                    if move is None:
                        done = True
                        break
                    new_board = board.copy()
                    new_board[move[0], move[1]] = opponent

                    win, _ = check_winner(new_board, opponent)
                    draw = is_draw(new_board)
                    reward = 0
                    if win:
                        reward = -20
                        lose_count += 1
                        done = True
                    elif draw:
                        reward = 0
                        done = True

                    if prev_state is not None and prev_action is not None:
                        self.update_q_value(prev_state, prev_action, reward, self.get_state_key(new_board))
                    board = new_board
                    current_player = self.player

            self.epsilon = max(0.01, self.epsilon * self.decay_factor)
            if (episode + 1) % 1000 == 0:
                print(f"Agent {self.player}: Episode {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.4f}")
        print(f"Final Win count: {win_count}, Loss Count: {lose_count}")


# Global Q-learning agent cache to avoid retraining repeatedly
q_agent_cache = {}  # key: ("X", num_episodes) or ("O", num_episodes)


def get_best_move_qlearning(board, player, episodes):
    global q_agent_cache
    key = (player, episodes)
    return q_agent_cache[key].choose_action(board, training=False)


# -----------------------------
# Default Opponent
# -----------------------------
def get_default_move(board, player):
    opponent = 'X' if player == 'O' else 'O'
    moves = get_valid_moves(board)
    for move in moves:
        temp_board = board.copy()
        temp_board[move[0], move[1]] = player
        if check_winner(temp_board, player)[0]:
            return move
    for move in moves:
        temp_board = board.copy()
        temp_board[move[0], move[1]] = opponent
        if check_winner(temp_board, opponent)[0]:
            return move
    return random.choice(moves) if moves else None


# -----------------------------
# TicTacToeGame Simulation
# -----------------------------
class TicTacToeGame:
    def __init__(self, algorithm_x, algorithm_o, animate=True):
        self.algorithm_x = algorithm_x
        self.algorithm_o = algorithm_o
        self.board = np.full((3, 3), ' ')
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
        else:
            return "Unknown Player"

    def play_game(self):
        self.frames.append(self.board.copy())
        self.moves_info.append("Game start")

        while not self.game_over:
            if self.current_player == 'X':
                move = self.get_move(self.algorithm_x, 'X')
            else:
                move = self.get_move(self.algorithm_o, 'O')

            if move is None:
                break

            self.board[move[0], move[1]] = self.current_player
            move_info = f"{self.current_player} moved to {move}"
            self.moves_info.append(move_info)
            self.frames.append(self.board.copy())

            win, cells = check_winner(self.board, self.current_player)
            if win:
                self.game_over = True
                self.winning_cells = cells
                self.result = self.current_player
            elif is_draw(self.board):
                self.game_over = True
                self.result = 'Draw'
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'

        if self.winning_cells:
            winner = self.current_player
            algo_name = self.get_algorithm_name(winner)
            self.moves_info.append(f"Winner: {algo_name} ({winner})")
            self.result = winner
        else:
            x_algo = self.get_algorithm_name('X')
            o_algo = self.get_algorithm_name('O')
            self.moves_info.append(f"Draw between {x_algo} (X) and {o_algo} (O)")
            self.result = 'Draw'
        self.frames.append(self.board.copy())
        if self.animate:
            self.animate_game()
        return self.result

    def get_move(self, algorithm_choice, player):
        if algorithm_choice == 1:
            return get_best_move_minimax(self.board, player, use_alpha_beta=False)
        elif algorithm_choice == 2:
            return get_best_move_minimax(self.board, player, use_alpha_beta=True)
        elif algorithm_choice in (3, 4, 5):
            ql_train_map = {3: 10000, 4: 20000, 5: 30000}
            episodes = ql_train_map[algorithm_choice]
            return get_best_move_qlearning(self.board, player, episodes)
        elif algorithm_choice == 6:
            return get_default_move(self.board, player)
        else:
            moves = get_valid_moves(self.board)
            return random.choice(moves) if moves else None

    def animate_game(self):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            board = self.frames[frame]
            title = self.moves_info[frame] if frame < len(self.moves_info) else ""
            highlight = self.winning_cells if frame == len(self.frames) - 1 else []
            self.draw_board(ax, board, highlight, title)
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=1000, repeat=False)
        plt.show()

    def draw_board(self, ax, board, highlight_cells=None, title=""):
        ax.set_xticks([0.5, 1.5, 2.5], minor=True)
        ax.set_yticks([0.5, 1.5, 2.5], minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(3):
            for j in range(3):
                content = board[i, j]
                color = 'red' if highlight_cells and (i, j) in highlight_cells else 'black'
                ax.text(j, 2 - i, content, fontsize=40, ha='center', va='center', color=color)
        ax.set_title(title)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)


def simulate_games(algorithm_x, algorithm_o, num_games=100):
    x_wins = 0
    o_wins = 0
    draws = 0
    total_time = 0.0
    total_moves = 0
    for _ in range(num_games):
        start_time = time.time()
        game = TicTacToeGame(algorithm_x, algorithm_o, animate=False)
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

        # Count moves as the number of non-empty cells on the final board.
        moves_count = np.count_nonzero(game.board != ' ')
        total_moves += moves_count

    avg_time = total_time / num_games
    avg_moves = total_moves / num_games
    return x_wins, o_wins, draws, avg_time, avg_moves


# -----------------------------
# Main Program
# -----------------------------
def startUp():
    global q_agent_cache
    print("Choose simulation (1: Run a single animated game, 2: Run batch simulations)")
    sim_choice = int(input().strip())

    ql_train_map = {3: 10000, 4: 20000, 5: 30000}

    if sim_choice == 1:
        print(
            "Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning(10k episodes), 4: Q-learning(20k "
            "episodes), 5: Q-learning(30k episodes), 6: Default): ")
        choice_x = int(input().strip())
        print(
            "Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning(10k episodes), 4: Q-learning(20k "
            "episodes), 5: Q-learning(30k episodes), 6: Default): ")
        choice_o = int(input().strip())

        if choice_x in (3, 4, 5):
            current_q_episode_x = ql_train_map[choice_x]
            key = ('X', current_q_episode_x)
            if key not in q_agent_cache:
                print(f"Training Q-learning agent for X with {current_q_episode_x} episodes...")
                q_agent_cache[key] = QLearningAgent('X')
                q_agent_cache[key].train(num_episodes=current_q_episode_x)
        if choice_o in (3, 4, 5):
            current_q_episode_o = ql_train_map[choice_o]
            key = ('O', current_q_episode_o)
            if key not in q_agent_cache:
                print(f"Training Q-learning agent for O with {current_q_episode_o} episodes...")
                q_agent_cache[key] = QLearningAgent('O')
                q_agent_cache[key].train(num_episodes=current_q_episode_o)

        TicTacToeGame(choice_x, choice_o, animate=True)
    else:
        for choice in (3, 4, 5):
            key_x = ('X', ql_train_map[choice])
            key_o = ('O', ql_train_map[choice])
            if key_x not in q_agent_cache:
                print(f"Pre-training Q-learning agent for X with {ql_train_map[choice]} episodes...")
                q_agent_cache[key_x] = QLearningAgent('X')
                q_agent_cache[key_x].train(num_episodes=ql_train_map[choice])
            if key_o not in q_agent_cache:
                print(f"Pre-training Q-learning agent for O with {ql_train_map[choice]} episodes...")
                q_agent_cache[key_o] = QLearningAgent('O')
                q_agent_cache[key_o].train(num_episodes=ql_train_map[choice])
        # Set number of games for each evaluation
        sim_num_games = 100

        # 1. Evaluation: Algorithms vs Default Opponent (Algorithm 6)
        eval_vs_default = {}
        for choice in range(1, 7):
            if choice == 6:  # Skip default vs default
                continue
            algo_name = get_algorithm_name(choice)
            print(f"Simulating {algo_name} (as X) vs Default Opponent (O)...")
            x_wins, o_wins, draws, avg_time, avg_moves = simulate_games(choice, 6, num_games=sim_num_games)
            eval_vs_default[algo_name] = (x_wins, o_wins, draws, avg_time, avg_moves)

        # Write CSV for vs Default evaluation
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

        # Graph for vs Default evaluation
        algos = list(eval_vs_default.keys())
        win_rates = [round((eval_vs_default[a][0] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]
        loss_rates = [round((eval_vs_default[a][1] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]
        draw_rates = [round((eval_vs_default[a][2] / (sum(eval_vs_default[a][:3])) * 100), 2) for a in algos]

        x = np.arange(len(algos))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, win_rates, width, label='Win Rate')
        ax.bar(x, loss_rates, width, label='Loss Rate')
        ax.bar(x + width, draw_rates, width, label='Draw Rate')
        ax.set_ylabel('Percentage')
        ax.set_title('Algorithms vs Default Opponent')
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.legend()
        plt.tight_layout()
        plt.savefig("vs_default_graph.png", bbox_inches='tight')
        plt.close()

        # 2. Evaluation: Head-to-Head (Only for algorithms 1 to 5)
        eval_head2head = {}
        for i in range(1, 6):
            for j in range(i + 1, 6):
                algo_i = get_algorithm_name(i)
                algo_j = get_algorithm_name(j)
                print(f"Simulating {algo_i} (X) vs {algo_j} (O)...")
                x_wins, o_wins, draws, avg_time, avg_moves = simulate_games(i, j, num_games=sim_num_games)
                eval_head2head[(algo_i, algo_j)] = (x_wins, o_wins, draws, avg_time, avg_moves)

        # Write CSV for head-to-head evaluation
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

        # Graph for head-to-head evaluation (Win rates)
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

        # Write CSV for overall evaluation
        overall_csv = "overall_results.csv"
        with open(overall_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Algorithm", "Overall Win Rate (%)", "Avg Game Time (s)", "Avg Moves"])
            for algo, (win_rate, avg_time_overall, avg_moves_overall) in overall_results.items():
                writer.writerow([algo, win_rate, f"{avg_time_overall:.4f}", f"{avg_moves_overall:.2f}"])
        print(f"Overall evaluation results saved to {overall_csv}")

        # Graph for overall evaluation
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


if __name__ == "__main__":
    startUp()
