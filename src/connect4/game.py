import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy
import math
import csv
from collections import defaultdict

# ---------------------------
# Helper functions for Connect4
# ---------------------------
def get_algorithm_name(choice):
    algorithm_names = {
        1: "Minimax",
        2: "Minimax+AB",
        3: "Q-learning",
        4: "Default"
    }
    return algorithm_names.get(choice, "Unknown Algorithm")

def count_winning_moves(board, player):
    count = 0
    for move in get_valid_moves(board):
        row = get_drop_row(board, move)
        temp_board = board.copy()
        temp_board[row, move] = player
        if check_winner(temp_board, player):
            count += 1
    return count

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
    # Check every position for a winning 4 in a row
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
    # Evaluation function for non-terminal states (used at depth limit)
    if check_winner(board, 'X'):
        return 1000
    elif check_winner(board, 'O'):
        return -1000
    else:
        return 0

# ---------------------------
# Role-Agnostic Minimax for Connect4 (Player Bias Fix)
# ---------------------------
def minimax(board, current_player, origin, use_alpha_beta=False, alpha=-math.inf, beta=math.inf, depth=0, depth_limit=4):
    opponent = 'O' if origin == 'X' else 'X'
    # Terminal condition: win/loss/draw or depth limit reached.
    if check_winner(board, origin):
        return 1000 - depth  # Sooner win is better
    elif check_winner(board, opponent):
        return -1000 + depth  # Delay loss if possible
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
# Q-Learning for Connect4 with Previous State/Action Update
# ---------------------------
class QLearningAgentConnect4:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=0.1, decay_factor=0.9999, player='X'):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.player = player
        # For handling previous state/action updates to mitigate bias:
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
        for episode in range(num_episodes):
            board = np.full((6, 7), ' ')
            current_player = self.player  # our agent's turn
            state = self.get_state_key(board)
            self.prev_state = None
            self.prev_action = None
            done = False
            opponent = 'O' if self.player == 'X' else 'X'

            while not done:
                if current_player == self.player:
                    # Compute pre-move block potential
                    pre_block = count_winning_moves(board, opponent)

                    action = self.choose_action(board, training=True)
                    if action is None:
                        break
                    row = get_drop_row(board, action)
                    board[row, action] = self.player
                    next_state = self.get_state_key(board)

                    # Terminal condition checks
                    if check_winner(board, self.player):
                        reward = 1
                        done = True
                    elif is_draw(board):
                        reward = 0
                        done = True
                    else:
                        # Compute post-move block potential
                        post_block = count_winning_moves(board, opponent)
                        reward = 0.1 * (pre_block - post_block)
                        done = False

                    # Update Q-value for previous move (if exists)
                    if self.prev_state is not None and self.prev_action is not None:
                        self.update_q_value(self.prev_state, self.prev_action, reward, next_state)
                    self.prev_state = state
                    self.prev_action = action
                    state = next_state
                    current_player = opponent  # switch turn

                else:
                    # Opponent's move: use default strategy
                    action = get_default_move(board, opponent)
                    if action is None:
                        break
                    row = get_drop_row(board, action)
                    board[row, action] = opponent
                    if check_winner(board, opponent):
                        reward = -1
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
                print(f"Q-Learning Connect4 Agent ({self.player}) Episode {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.4f}")

# Use separate agents for X and O to avoid bias.
q_learning_agent_connect4_X = None
q_learning_agent_connect4_O = None

def get_best_move_qlearning(board, player):
    global q_learning_agent_connect4_X, q_learning_agent_connect4_O
    if player == 'X':
        if q_learning_agent_connect4_X is None:
            q_learning_agent_connect4_X = QLearningAgentConnect4(player='X')
            print("Training Q-learning agent for Connect4 as X...")
            q_learning_agent_connect4_X.train(num_episodes=10000)
        return q_learning_agent_connect4_X.choose_action(board, training=False)
    else:
        if q_learning_agent_connect4_O is None:
            q_learning_agent_connect4_O = QLearningAgentConnect4(player='O')
            print("Training Q-learning agent for Connect4 as O...")
            q_learning_agent_connect4_O.train(num_episodes=10000)
        return q_learning_agent_connect4_O.choose_action(board, training=False)

# ---------------------------
# Default Opponent
# ---------------------------
def get_default_move(board, player):
    # Default opponent: if winning move exists, take it; else block; else random.
    opponent = 'X' if player == 'O' else 'O'
    moves = get_valid_moves(board)
    for move in moves:
        temp_board = board.copy()
        row = get_drop_row(temp_board, move)
        temp_board[row, move] = player
        if check_winner(temp_board, player):
            return move
    for move in moves:
        temp_board = board.copy()
        row = get_drop_row(temp_board, move)
        temp_board[row, move] = opponent
        if check_winner(temp_board, opponent):
            return move
    return random.choice(moves) if moves else None

# ---------------------------
# Connect4Game Simulation with Animation and CSV Logging
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
        else:
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
            winner_algo = self.get_algorithm_name(self.result)
            self.moves_info.append(f"Winner: {winner_algo} ({self.result})")
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
            return get_best_move_minimax(self.board, player, use_alpha_beta=False)
        elif algorithm_choice == 2:
            return get_best_move_minimax(self.board, player, use_alpha_beta=True)
        elif algorithm_choice == 3:
            return get_best_move_qlearning(self.board, player)
        elif algorithm_choice == 4:
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
        if game.result == 'X':
            x_wins += 1
        elif game.result == 'O':
            o_wins += 1
        else:
            draws += 1
    return x_wins, o_wins, draws

# ---------------------------
# Main Program
# ---------------------------
def startUp():
    # Pre-train Q-learning agents for both players
    global q_learning_agent_connect4_X, q_learning_agent_connect4_O
    q_learning_agent_connect4_X = QLearningAgentConnect4(player='X')
    q_learning_agent_connect4_O = QLearningAgentConnect4(player='O')
    print("Training Q-learning agent for Connect4 as X...")
    q_learning_agent_connect4_X.train(num_episodes=10000)
    print("Training Q-learning agent for Connect4 as O...")
    q_learning_agent_connect4_O.train(num_episodes=10000)

    print("Choose simulation (1: Run a single animated game, 2: Run batch simulations)")
    sim_choice = int(input().strip())
    if sim_choice == 1:
        print("Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning, 4: Default): ")
        choice_x = int(input().strip())
        print("Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning, 4: Default): ")
        choice_o = int(input().strip())
        Connect4Game(choice_x, choice_o, animate=True)
    else:
        results = {}
        simulation_games = 20  # Number of games per pairing
        pair_labels = []
        x_wins_list = []
        o_wins_list = []
        draws_list = []

        for choice_x in [1, 2, 3, 4]:
            for choice_o in [1, 2, 3, 4]:
                if choice_x == choice_o:
                    continue
                x_algo = get_algorithm_name(choice_x)
                o_algo = get_algorithm_name(choice_o)
                print(f"Simulating {x_algo} (X) vs {o_algo} (O) for {simulation_games} games...")
                x_wins, o_wins, draws = simulate_games(choice_x, choice_o, simulation_games)
                results[(x_algo, o_algo)] = (x_wins, o_wins, draws)
                pair_labels.append(f"{x_algo} vs {o_algo}")
                x_wins_list.append(x_wins)
                o_wins_list.append(o_wins)
                draws_list.append(draws)
                print("Results:")
                print(f"{x_algo} (X) wins: {x_wins}")
                print(f"{o_algo} (O) wins: {o_wins}")
                print(f"Draws: {draws}")
                print("------")

        csv_filename = "connect4_results.csv"
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Pairing", "X Wins", "O Wins", "Draws"])
            for pair, (x_wins, o_wins, draws) in results.items():
                pairing = f"{pair[0]} vs {pair[1]}"
                print("{:<20} {:<10} {:<10} {:<10}".format(pairing, x_wins, o_wins, draws))
                writer.writerow([pairing, x_wins, o_wins, draws])
        print(f"\nResults saved to {csv_filename}")

        labels = pair_labels
        x_axis = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots()
        rects1 = ax.bar(x_axis - width, x_wins_list, width, label='X wins')
        rects2 = ax.bar(x_axis, o_wins_list, width, label='O wins')
        rects3 = ax.bar(x_axis + width, draws_list, width, label='Draws')

        ax.set_ylabel('Number of Wins/Draws')
        ax.set_title('Connect4 Simulation Results by Pairing')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()
        png_filename = "connect4_final.png"
        plt.savefig(png_filename)
        print(f"Grouped bar chart saved as {png_filename}")
        plt.close()

if __name__ == "__main__":
    startUp()
