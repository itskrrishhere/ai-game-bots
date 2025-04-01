import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy
import math
from collections import defaultdict


# -----------------------------
# Helper functions for Tic Tac Toe
# -----------------------------
def get_algorithm_name(choice):
    algorithm_names = {
        1: "Minimax", 2: "Minimax+AB",
        3: "Q-learning", 4: "Default"
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


def evaluate(board):
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


def get_best_move_minimax(board, player, use_alpha_beta=False):
    moves = get_valid_moves(board)
    best_move = None
    if not moves:
        return None

    maximizing = (player == 'X')
    best_score = -math.inf if maximizing else math.inf

    for move in moves:
        new_board = board.copy()
        new_board[move[0], move[1]] = player
        score = minimax(new_board, 0, not maximizing, use_alpha_beta, -math.inf, math.inf)
        if (maximizing and score > best_score) or (not maximizing and score < best_score):
            best_score = score
            best_move = move
    return best_move


# -----------------------------
# Q-Learning
# -----------------------------
class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, decay_factor=0.9999):
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
        if next_state is None:
            next_max_q = 0.0
        else:
            next_max_q = max([self.get_q_value(next_state, a) for a in get_valid_moves(np.array(next_state))] or [0.0])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, board, player, training=True):
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

    def train(self, num_episodes=10000, opponent_algorithm=4):
        for episode in range(num_episodes):
            board = np.full((3, 3), ' ')
            current_player = 'X'
            prev_state = None
            prev_action = None
            done = False

            while not done:
                if current_player == 'X':
                    state = self.get_state_key(board)
                    valid_moves = get_valid_moves(board)
                    if not valid_moves:
                        break
                    action = self.choose_action(board, 'X', training=True)
                    if action is None:
                        break
                    new_board = board.copy()
                    new_board[action] = 'X'

                    win_X, _ = check_winner(new_board, 'X')
                    draw = is_draw(new_board)
                    reward = 0
                    if win_X:
                        reward = 1
                        done = True
                    elif draw:
                        reward = 0
                        done = True
                    else:
                        done = False

                    if prev_state is not None and prev_action is not None:
                        self.update_q_value(prev_state, prev_action, reward, self.get_state_key(new_board))

                    prev_state = state
                    prev_action = action
                    board = new_board
                    current_player = 'O'

                    if done:
                        self.update_q_value(state, action, reward, None)
                else:
                    move = get_default_move(board, 'O')
                    if move is None:
                        done = True
                        reward = 0
                        break
                    new_board = board.copy()
                    new_board[move] = 'O'

                    win_O, _ = check_winner(new_board, 'O')
                    draw = is_draw(new_board)
                    if win_O:
                        reward = -1
                        done = True
                    elif draw:
                        reward = 0
                        done = True
                    else:
                        done = False

                    if prev_state is not None and prev_action is not None:
                        next_state = self.get_state_key(new_board)
                        self.update_q_value(prev_state, prev_action, reward, next_state)

                    board = new_board
                    current_player = 'X'

                    if done:
                        break

            self.epsilon = max(0.01, self.epsilon * self.decay_factor)

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.4f}")


q_learning_agent = None


def get_best_move_qlearning(board, player):
    global q_learning_agent
    if q_learning_agent is None:
        q_learning_agent = QLearningAgent()
        print("Training Q-learning agent. This may take a few minutes...")
        q_learning_agent.train(num_episodes=10000)
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return None
    action = q_learning_agent.choose_action(board, player, training=False)
    return action


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
        algorithm_names = {
            1: "Minimax", 2: "Minimax+AB",
            3: "Q-learning", 4: "Default"
        }
        if player == 'X':
            return algorithm_names.get(self.algorithm_x, "Unknown Algorithm")
        elif player == 'O':
            return algorithm_names.get(self.algorithm_o, "Unknown Algorithm")
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
            winner = 'X' if check_winner(self.board, 'X')[0] else 'O'
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
        elif algorithm_choice == 3:
            return get_best_move_qlearning(self.board, player)
        elif algorithm_choice == 4:
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
                color = 'red' if (i, j) in highlight_cells else 'black'
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
    global q_learning_agent
    q_learning_agent = QLearningAgent()
    print("Training Q-learning agent...")
    q_learning_agent.train(num_episodes=10000)
    print("Choose simulation (1: Run a single animated game, 2: run batch(20) simulations)")
    sim_choice = int(input().strip())
    if sim_choice == 1:
        # To Run a single animated game
        print("Choose algorithm for X (1: Minimax, 2: Minimax+AB, 3: Q-learning, 4: Default): ")
        choice_x = int(input().strip())
        print("Choose algorithm for O (1: Minimax, 2: Minimax+AB, 3: Q-learning, 4: Default): ")
        choice_o = int(input().strip())
        TicTacToeGame(choice_x, choice_o, animate=True)
    else:
        #  To run simulations and print results
        results = {}
        simulation_games = 20 # Number of games per pairing
        pair_labels = []
        x_wins_list = []
        o_wins_list = []
        draws_list = []

        # Simulate distinct pairings (choice_x from 1 to 3 and choice_o from choice_x+1 to 4)
        for choice_x in range(1, 4):
            for choice_o in range(choice_x + 1, 5):
                x_algo = get_algorithm_name(choice_x)
                o_algo = get_algorithm_name(choice_o)
                print(f"Simulating {x_algo} (X) vs {o_algo} (O) for {simulation_games} games...")
                x_wins, o_wins, draws = simulate_games(choice_x, choice_o, simulation_games)
                results[(x_algo, o_algo)] = (x_wins, o_wins, draws)
                pair_labels.append(f"{x_algo} vs {o_algo}")
                x_wins_list.append(x_wins)
                o_wins_list.append(o_wins)
                draws_list.append(draws)
                print(f"Results:")
                print(f"{x_algo} (X) wins: {x_wins}")
                print(f"{o_algo} (O) wins: {o_wins}")
                print(f"Draws: {draws}")
                print("------")

        # Tabulate results
        print("\nFinal Results Tabulation:")
        print("{:<20} {:<10} {:<10} {:<10}".format("Pairing", "X wins", "O wins", "Draws"))
        for pair, (x_wins, o_wins, draws) in results.items():
            print("{:<20} {:<10} {:<10} {:<10}".format(f"{pair[0]} vs {pair[1]}", x_wins, o_wins, draws))

        # Create a grouped bar chart for the results
        labels = pair_labels
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, x_wins_list, width, label='X wins')
        rects2 = ax.bar(x, o_wins_list, width, label='O wins')
        rects3 = ax.bar(x + width, draws_list, width, label='Draws')

        ax.set_ylabel('Number of Wins/Draws')
        ax.set_title('Simulation Results by Pairing')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    startUp()
