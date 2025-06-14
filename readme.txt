# 🧠 AI Game Bots

This project implements AI agents to play **Tic-Tac-Toe** and **Connect4**, with support for multiple algorithms, batch simulations, and visual performance evaluation.

---

## 🎮 Features

* ✅ Play or simulate games of **Tic-Tac-Toe** and **Connect4**
* 🧠 Compete using different algorithms:

  * Minimax
  * Minimax with Alpha-Beta Pruning
  * Q-learning (at various training levels)
  * Default (Random or Rule-based)
* 🧪 Batch simulation and evaluation:

  * Algorithms vs Default Opponent
  * Head-to-Head between algorithms
  * Aggregated performance comparisons
* 📊 Output results as:

  * **CSV files** for data
  * **PNG graphs** for visualization

---

## 📁 Directory Structure

```
ai-game-bots/
├── connect4/
│   ├── vs_default_results.csv
│   ├── vs_default_graph.png
│   ├── head2head_results.csv
│   ├── head2head_graph.png
│   ├── overall_results.csv
│   └── overall_graph.png
├── tictactoe/
│   └── (same as above)
├── game.py
├── agent_qlearning.py
├── agent_minimax.py
├── utils.py
└── main.py
```

---

## 🚀 How to Run

### 🧰 Install Dependencies

Ensure you have **Python 3.9+** and install the following packages:

```bash
pip install numpy matplotlib
```

### ▶️ Run the Game

```bash
python main.py
```

Follow the prompts:

1. Choose the game: **Tic-Tac-Toe** or **Connect4**
2. Choose simulation type:

   * Single Game
   * Batch Simulations
3. Select algorithms for **X** and **O**
4. View and analyze results (CSV + Graphs)

---

## 📂 Output Files

Each game (Connect4, Tic-Tac-Toe) will generate:

| File                     | Description                                 |
| ------------------------ | ------------------------------------------- |
| `vs_default_results.csv` | Win rates vs default opponent               |
| `vs_default_graph.png`   | Bar graph of above results                  |
| `head2head_results.csv`  | Round-robin results for all algorithm pairs |
| `head2head_graph.png`    | Graphical visualization of head-to-head     |
| `overall_results.csv`    | Aggregated performance summary              |
| `overall_graph.png`      | Visual summary of all algorithms            |

---

## 🛠️ Customization

* To add new algorithms, **extend the logic** in:

  * `game.py` (game logic)
  * Corresponding agent files
* Q-learning agents can be tweaked for:

  * Training episodes
  * Learning rate
  * Exploration strategy

---

## 📜 License

This project is intended for **educational and research purposes only**.

---
