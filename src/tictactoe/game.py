import csv
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from collections import defaultdict


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
# Minimax Algorithms (Role-Agnostic)
# -----------------------------
def minimax(board, current_player, origin, use_alpha_beta=False, alpha=-math.inf, beta=math.inf):
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
        return best_score
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
# Q-Learning (Role-Agnostic Version)
# -----------------------------
class QLearningAgent:
    def __init__(self, player, alpha=0.5, gamma=0.9, epsilon=0.1, decay_factor=0.9999):
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
        temp_board = board.copy()
        # Check if opponent would win on their next move if we don't take this action
        for move in get_valid_moves(board):
            test_board = board.copy()
            test_board[move[0], move[1]] = opponent
            if check_winner(test_board, opponent)[0]:
                temp_board[action[0], action[1]] = self.player
                test_board = temp_board.copy()
                test_board[move[0], move[1]] = opponent
                if not check_winner(test_board, opponent)[0]:
                    return True
        return False

    def train(self, num_episodes=10000):
        opponent = 'O' if self.player == 'X' else 'X'
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
                        reward = 10
                        done = True
                    elif draw:
                        reward = 0
                        done = True
                    elif self.blocks_opponent(board, action):
                        reward = 1

                    if prev_state is not None and prev_action is not None:
                        self.update_q_value(prev_state, prev_action, reward, self.get_state_key(new_board))
                    prev_state = state
                    prev_action = action
                    board = new_board
                    current_player = opponent

                    if done:
                        self.update_q_value(state, action, reward, None)
                else:
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
                        reward = -10
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
        # For Q-learning options (choices 3,4,5), pass the corresponding episode count.
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
    for _ in range(num_games):
        game = TicTacToeGame(algorithm_x, algorithm_o, animate=False)
        result = game.result
        if result == 'X':
            x_wins += 1
        elif result == 'O':
            o_wins += 1
        else:
            draws += 1
    return x_wins, o_wins, draws


# -----------------------------
# Main Program
# -----------------------------
def startUp():
    global q_agent_cache, current_q_episode_X, current_q_episode_O
    # Ask user for simulation type
    print("Choose simulation (1: Run a single animated game, 2: Run batch simulations)")
    sim_choice = int(input().strip())

    # Mapping for Q-learning training episodes
    ql_train_map = {3: 10000, 4: 20000, 5: 30000}

    if sim_choice == 1:
        print(
            "Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning(10k episodes), 4: Q-learning(20k episodes), 5: Q-learning(30k episodes), 6: Default): ")
        choice_x = int(input().strip())
        print(
            "Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning(10k episodes), 4: Q-learning(20k episodes), 5: Q-learning(30k episodes), 6: Default): ")
        choice_o = int(input().strip())

        # For Q-learning, use caching to train agents only once.
        if choice_x in (3, 4, 5):
            current_q_episode_X = ql_train_map[choice_x]
            key = ('X', current_q_episode_X)
            if key not in q_agent_cache:
                print(f"Training Q-learning agent for X with {current_q_episode_X} episodes...")
                q_agent_cache[key] = QLearningAgent('X')
                q_agent_cache[key].train(num_episodes=current_q_episode_X)
        if choice_o in (3, 4, 5):
            current_q_episode_O = ql_train_map[choice_o]
            key = ('O', current_q_episode_O)
            if key not in q_agent_cache:
                print(f"Training Q-learning agent for O with {current_q_episode_O} episodes...")
                q_agent_cache[key] = QLearningAgent('O')
                q_agent_cache[key].train(num_episodes=current_q_episode_O)

        TicTacToeGame(choice_x, choice_o, animate=True)
    else:
        # In batch simulation, pretrain the Q-learning agents for choices 3,4,5 once and reuse them.
        for choice in (3, 4, 5):
            key_X = ('X', ql_train_map[choice])
            key_O = ('O', ql_train_map[choice])
            if key_X not in q_agent_cache:
                print(f"Pre-training Q-learning agent for X with {ql_train_map[choice]} episodes...")
                q_agent_cache[key_X] = QLearningAgent('X')
                q_agent_cache[key_X].train(num_episodes=ql_train_map[choice])
            if key_O not in q_agent_cache:
                print(f"Pre-training Q-learning agent for O with {ql_train_map[choice]} episodes...")
                q_agent_cache[key_O] = QLearningAgent('O')
                q_agent_cache[key_O].train(num_episodes=ql_train_map[choice])
        for num_games in (20,40,60,80):
            results = {}
            pair_labels = []
            x_win_rates = []
            o_win_rates = []
            draw_rates = []

            # Loop over algorithm choices from 1 to 6
            for choice_x in range(1, 7):
                for choice_o in range(choice_x + 1, 7):
                    # Skip simulation when both agents are Q-learning to avoid pairing same Q-learning setups repeatedly
                    if choice_x in (3, 4, 5) and choice_o in (3, 4, 5):
                        continue
                    x_algo = get_algorithm_name(choice_x)
                    o_algo = get_algorithm_name(choice_o)
                    print(f"Simulating {x_algo} (X) vs {o_algo} (O) ...")
                    x_wins, o_wins, draws = simulate_games(choice_x, choice_o, num_games=num_games)
                    x_win_rate = round((x_wins / num_games) * 100, 2)
                    o_win_rate = round((o_wins / num_games) * 100, 2)
                    draw_rate = round((draws / num_games) * 100, 2)
                    results[(x_algo, o_algo)] = (x_win_rate, o_win_rate, draw_rate)
                    pair_labels.append(f"{x_algo} vs {o_algo}")
                    x_win_rates.append(x_win_rate)
                    o_win_rates.append(o_win_rate)
                    draw_rates.append(draw_rate)
                    print(f"{x_algo} win rate: {x_win_rate}%")
                    print(f"{o_algo} win rate: {o_win_rate}%")
                    print(f"Draw rate: {draw_rate}%")
                    print("------")

            folder_name = str(num_games)

            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)  # Delete folder if it exists

            os.mkdir(folder_name)  # Create new folder
            print(f"Folder '{folder_name}' created successfully!")

            print("\nFinal Results Tabulation:")
            csv_filename = f"{num_games}\\tictactoe_results.csv"
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["Pairing (X VS O)", "X Algorithm Win Rate (%)", "O Algorithm Win Rate (%)", "Draw Rate (%)"])
                for pair, (x_win_rate, o_win_rate, draw_rate) in results.items():
                    pairing = f"{pair[0]} vs {pair[1]}"
                    print("{:<20} {:<10} {:<10} {:<10}".format(pairing, x_win_rate, o_win_rate, draw_rate))
                    writer.writerow([pairing, x_win_rate, o_win_rate, draw_rate])
            print(f"\nResults saved to {csv_filename}")

            x_vals = np.arange(len(pair_labels))
            width = 0.1
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x_vals - width, x_win_rates, width, label='X Algorithm win rate')
            ax.bar(x_vals, o_win_rates, width, label='O Algorithm win rate')
            ax.bar(x_vals + width, draw_rates, width, label='Draw rate')
            ax.set_ylabel('Win Rate (%)')
            ax.set_title('Simulation Results by Pairing')
            ax.set_xticks(x_vals)
            ax.set_xticklabels(pair_labels, rotation=45, ha='right')

            # Place legend outside the plot on the right
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

            fig.tight_layout()
            # Save with bbox_inches='tight' to include the legend
            plt.savefig(f"{num_games}\\tictactoe_graph.png", bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    startUp()
