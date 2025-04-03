import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import csv
import os
import shutil
from collections import defaultdict

# ---------------------------
# Helper Functions for Connect4
# ---------------------------
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
    # Valid moves are the columns (0-6) with at least one empty cell at the top.
    return [col for col in range(7) if board[0, col] == ' ']

def get_drop_row(board, col):
    # Returns the lowest available row in column 'col'
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
            return []  # Out of bounds
        cells.append((r, c))
    if all(board[r, c] == player for r, c in cells):
        return cells
    return []

def check_winner(board, player):
    # Check every board cell for a winning line
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
    # A basic evaluation function when depth limit is reached.
    if check_winner(board, 'X'):
        return 1000
    elif check_winner(board, 'O'):
        return -1000
    else:
        return 0

# ---------------------------
# Minimax for Connect4 (Role-Agnostic)
# ---------------------------
def minimax(board, current_player, origin, use_alpha_beta=False, alpha=-math.inf, beta=math.inf, depth=0, depth_limit=4):
    opponent = 'O' if origin == 'X' else 'X'
    if check_winner(board, origin):
        return 1000 - depth  # sooner win is better
    elif check_winner(board, opponent):
        return -1000 + depth
    elif is_draw(board) or depth == depth_limit:
        return evaluate(board)

    valid_moves = get_valid_moves(board)
    if current_player == origin:
        best_score = -math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score = minimax(new_board, opponent, origin, use_alpha_beta, alpha, beta, depth + 1, depth_limit)
            best_score = max(best_score, score)
            if use_alpha_beta:
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = math.inf
        for col in valid_moves:
            row = get_drop_row(board, col)
            if row is None:
                continue
            new_board = board.copy()
            new_board[row, col] = current_player
            score = minimax(new_board, origin, origin, use_alpha_beta, alpha, beta, depth + 1, depth_limit)
            best_score = min(best_score, score)
            if use_alpha_beta:
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score

def get_best_move_minimax(board, player, use_alpha_beta=False, depth_limit=4):
    valid_moves = get_valid_moves(board)
    best_move = None
    if not valid_moves:
        return None
    best_score = -math.inf if player == 'X' else math.inf
    for col in valid_moves:
        row = get_drop_row(board, col)
        if row is None:
            continue
        new_board = board.copy()
        new_board[row, col] = player
        opponent = 'O' if player == 'X' else 'X'
        score = minimax(new_board, opponent, player, use_alpha_beta, -math.inf, math.inf, 0, depth_limit)
        if (player == 'X' and score > best_score) or (player == 'O' and score < best_score):
            best_score = score
            best_move = col
    return best_move

# ---------------------------
# Q-Learning for Connect4 (Role-Agnostic)
# ---------------------------
class QLearningAgentConnect4:
    def __init__(self, player, alpha=0.5, gamma=0.99, epsilon=0.1, decay_factor=0.9999):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.player = player
        self.prev_state = None
        self.prev_action = None

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

    def train(self, num_episodes=10000):
        opponent = 'O' if self.player == 'X' else 'X'
        for episode in range(num_episodes):
            board = np.full((6, 7), ' ')
            current_player = self.player
            state = self.get_state_key(board)
            self.prev_state = None
            self.prev_action = None
            done = False
            while not done:
                if current_player == self.player:
                    action = self.choose_action(board, training=True)
                    if action is None:
                        break
                    row = get_drop_row(board, action)
                    board[row, action] = self.player
                    next_state = self.get_state_key(board)
                    if check_winner(board, self.player):
                        reward = 10
                        done = True
                    elif is_draw(board):
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = False
                    if self.prev_state is not None and self.prev_action is not None:
                        self.update_q_value(self.prev_state, self.prev_action, reward, next_state)
                    self.prev_state = state
                    self.prev_action = action
                    state = next_state
                    current_player = opponent
                    if done:
                        self.update_q_value(state, action, reward, None)
                else:
                    # Opponent uses default move
                    action = get_default_move(board, opponent)
                    if action is None:
                        break
                    row = get_drop_row(board, action)
                    board[row, action] = opponent
                    if check_winner(board, opponent):
                        reward = -10
                        done = True
                        self.update_q_value(state, self.prev_action, reward, self.get_state_key(board))
                    elif is_draw(board):
                        reward = 0
                        done = True
                    else:
                        done = False
                    current_player = self.player
            self.epsilon = max(0.01, self.epsilon * self.decay_factor)
            if (episode + 1) % 1000 == 0:
                print(f"Connect4 Q-learning Agent ({self.player}) Episode {episode+1}/"
                      f"{num_episodes}, Epsilon: {self.epsilon:.4f}")

# Global cache for Connect4 Q-learning agents: key -> (player, episodes)
q_agent_cache_connect4 = {}

def get_best_move_qlearning(board, player, episodes):
    global q_agent_cache_connect4
    key = (player, episodes)
    if key not in q_agent_cache_connect4:
        print(f"Training Q-learning agent for Connect4 as {player} with {episodes} episodes...")
        agent = QLearningAgentConnect4(player)
        agent.train(num_episodes=episodes)
        q_agent_cache_connect4[key] = agent
    return q_agent_cache_connect4[key].choose_action(board, training=False)

# ---------------------------
# Default Opponent for Connect4
# ---------------------------
def get_default_move(board, player):
    opponent = 'X' if player == 'O' else 'O'
    moves = get_valid_moves(board)
    # First, check for a winning move
    for move in moves:
        temp_board = board.copy()
        row = get_drop_row(temp_board, move)
        temp_board[row, move] = player
        if check_winner(temp_board, player):
            return move
    # Then, block opponent's winning move
    for move in moves:
        temp_board = board.copy()
        row = get_drop_row(temp_board, move)
        temp_board[row, move] = opponent
        if check_winner(temp_board, opponent):
            return move
    return random.choice(moves) if moves else None

# ---------------------------
# Connect4Game Simulation with Animation and Logging
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
        # Mapping for Q-learning episodes similar to Tic Tac Toe
        ql_train_map = {3: 10000, 4: 20000, 5: 30000}

        while not self.game_over:
            if self.current_player == 'X':
                move = self.get_move(self.algorithm_x, 'X', ql_train_map)
            else:
                move = self.get_move(self.algorithm_o, 'O', ql_train_map)
            if move is None:
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

    def get_move(self, algorithm_choice, player, ql_train_map):
        if algorithm_choice == 1:
            return get_best_move_minimax(self.board, player, use_alpha_beta=False)
        elif algorithm_choice == 2:
            return get_best_move_minimax(self.board, player, use_alpha_beta=True)
        elif algorithm_choice in (3, 4, 5):
            episodes = ql_train_map[algorithm_choice]
            return get_best_move_qlearning(self.board, player, episodes)
        elif algorithm_choice == 6:
            return get_default_move(self.board, player)
        else:
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
        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=1000, repeat=False)
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
# Simulation for Connect4
# ---------------------------
def simulate_games(algorithm_x, algorithm_o, num_games=20):
    x_wins = 0
    o_wins = 0
    draws = 0
    for _ in range(num_games):
        game = Connect4Game(algorithm_x, algorithm_o, animate=False)
        result = game.result
        if result == 'X':
            x_wins += 1
        elif result == 'O':
            o_wins += 1
        else:
            draws += 1
    return x_wins, o_wins, draws

# ---------------------------
# Main Program
# ---------------------------
def startUp():
    # Ask user for simulation type
    print("Choose simulation (1: Run a single animated game, 2: Run batch simulations)")
    sim_choice = int(input().strip())
    # Mapping for Q-learning episodes
    ql_train_map = {3: 10000, 4: 20000, 5: 30000}

    if sim_choice == 1:
        print("Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning (10k episodes), 4: Q-learning (20k episodes), 5: Q-learning (30k episodes), 6: Default):")
        choice_x = int(input().strip())
        print("Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning (10k episodes), 4: Q-learning (20k episodes), 5: Q-learning (30k episodes), 6: Default):")
        choice_o = int(input().strip())
        # Pre-train Q-learning agents if necessary
        for choice, player in [(choice_x, 'X'), (choice_o, 'O')]:
            if choice in (3, 4, 5):
                episodes = ql_train_map[choice]
                key = (player, episodes)
                if key not in q_agent_cache_connect4:
                    print(f"Training Q-learning agent for {player} with {episodes} episodes...")
                    agent = QLearningAgentConnect4(player)
                    agent.train(num_episodes=episodes)
                    q_agent_cache_connect4[key] = agent
        Connect4Game(choice_x, choice_o, animate=True)
    else:
        # In batch simulation, pretrain Q-learning agents for choices 3,4,5
        for choice in (3, 4, 5):
            for player in ('X', 'O'):
                key = (player, ql_train_map[choice])
                if key not in q_agent_cache_connect4:
                    print(f"Pre-training Q-learning agent for {player} with {ql_train_map[choice]} episodes...")
                    agent = QLearningAgentConnect4(player)
                    agent.train(num_episodes=ql_train_map[choice])
                    q_agent_cache_connect4[key] = agent

        simulation_games = 1  # Number of games per pairing
        results = {}
        pair_labels = []
        x_wins_list = []
        o_wins_list = []
        draws_list = []

        # Loop over algorithm choices (1 to 6) for X and O
        for choice_x in range(1, 7):
            for choice_o in range(choice_x + 1, 7):
                # For Q-learning pairings, avoid pairing same setups twice if desired.
                x_algo = get_algorithm_name(choice_x)
                o_algo = get_algorithm_name(choice_o)
                print(f"Simulating {x_algo} (X) vs {o_algo} (O) for {simulation_games} games...")
                x_wins, o_wins, draws = simulate_games(choice_x, choice_o, simulation_games)
                results[(x_algo, o_algo)] = (x_wins, o_wins, draws)
                pair_labels.append(f"{x_algo} vs {o_algo}")
                x_wins_list.append(x_wins)
                o_wins_list.append(o_wins)
                draws_list.append(draws)
                print(f"{x_algo} (X) wins: {x_wins}")
                print(f"{o_algo} (O) wins: {o_wins}")
                print(f"Draws: {draws}")
                print("------")

        folder_name = "connect4_simulation_results"
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)
        csv_filename = os.path.join(folder_name, "connect4_results.csv")
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Pairing (X vs O)", "X Wins", "O Wins", "Draws"])
            for pair, (x_wins, o_wins, draws) in results.items():
                pairing = f"{pair[0]} vs {pair[1]}"
                writer.writerow([pairing, x_wins, o_wins, draws])
        print(f"Results saved to {csv_filename}")

        # Plot grouped bar chart of the results
        x_axis = np.arange(len(pair_labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_axis - width, x_wins_list, width, label='X Wins')
        ax.bar(x_axis, o_wins_list, width, label='O Wins')
        ax.bar(x_axis + width, draws_list, width, label='Draws')
        ax.set_ylabel('Number of Wins/Draws')
        ax.set_title('Connect4 Simulation Results by Pairing')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        png_filename = os.path.join(folder_name, "connect4_graph.png")
        plt.savefig(png_filename, bbox_inches='tight')
        plt.close()
        print(f"Graph saved as {png_filename}")

if __name__ == "__main__":
    startUp()
