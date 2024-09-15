# Mastermind Game Solver - AI Project

This repository contains a project to solve the Mastermind game using two AI approaches: **Q-Learning** and **Genetic Algorithm**. Additionally, the project includes an interactive **Mastermind GUI** implemented in HTML, allowing users to play the game against an AI agent or manually.

## Mastermind Game Overview

Mastermind is a classic code-breaking game where a player attempts to guess a secret code made up of colored pegs. After each guess, the player receives feedback:
- **Black pegs** indicate correct colors in the correct positions.
- **White pegs** indicate correct colors in the wrong positions.

The goal is to guess the secret code within a limited number of attempts. In this project:
- The **Q-Learning agent** learns through trial and error, exploring different solutions.
- The **Genetic Algorithm** optimizes the solution by simulating natural selection, crossover, and mutation.

## Project Components

1. **Q-Learning Algorithm**: A reinforcement learning agent that learns to solve the Mastermind game through exploration and exploitation.
2. **Genetic Algorithm**: An evolutionary approach that optimizes the solution to the Mastermind game using natural selection principles.
3. **Output Directory**: Contains plots, CSV files, and other data generated during the training of the algorithms.
4. **Mastermind GUI**: HTML files to allow interactive play against a genetic algorithm-based AI or manually.

## Project Structure

- `Q_learning.py`: Python script implementing the Q-Learning algorithm for the Mastermind game.
- `genetic_algorithm.py`: Python script implementing the Genetic Algorithm to solve the game.
- `output/`: Directory containing:
  - **Plots**: Visual representations of the learning performance and results.
  - **CSV Files**: Data generated during the algorithm training, such as metrics for the number of guesses, time taken, and hyperparameter tuning.
- `GUI_Mastermind/`: Directory containing HTML files for the GUI that can be run locally in a browser. This includes options to play the game manually or against the AI.

## How to Run the Project

### 1. Running the Q-Learning Algorithm
You can run the Q-Learning algorithm by executing `Q_learning.py`. There are several configurations and simulations you can run:

- To simulate games and evaluate the agent’s performance over 1000 games:
  ```python
  simulate_games(epsilon=EPSILON, num_games=1000, alpha=ALPHA, discount=DISCOUNT,
                 max_guesses=MAX_GUESSES, code_length=genetic_algorithm.DEFAULT_C0DE_LENGTH, num_colors=genetic_algorithm.DEFAULT_NUM_COLORS)
  ```

  This will print the average time taken for each game, the number of turns taken, and the total training time.

- To find the best hyperparameter combination:
  ```python
  generate_hyperparameter_table(code_length=genetic_algorithm.DEFAULT_C0DE_LENGTH, num_colors=genetic_algorithm.DEFAULT_NUM_COLORS, num_of_games=1000)
  ```

  The results will be saved in the `output/` directory and the best combination will be printed to the screen.

- To generate performance heatmaps across different configurations of colors and positions:
  ```python
  colors_vs_positions_multithreaded(num_of_games=50, min_range_color=6, max_range_color=9, min_range_position=4, max_range_position=6)
  ```

  Heatmaps for average training time and turns will be generated and saved in the `output/` directory.

### 2. Running the Genetic Algorithm
To run the Genetic Algorithm, execute `genetic_algorithm.py` with the following configurations:

- Simulate 1000 games using the genetic algorithm:
  ```python
  simulate_games(num_simulations=1000, colors=DEFAULT_COLORS, code_length=DEFAULT_C0DE_LENGTH, pop_size=DEFAULT_MAX_POP_SIZE,
                 white_pegs_weight=DEFAULT_WHITE_PEGS_WEIGHT, black_pegs_weight=DEFAULT_BLACK_PEGS_WEIGHT,
                 initial_guess=DEFAULT_INITIAL_GUESS, elite_ratio=DEFAULT_ELITE_RATIO)
  ```

- To explore the impact of different weights on fitness:
  ```python
  different_weights()
  ```
  The results will be saved as `fitness_weights_results.csv` in the `output/` directory and the best combination will be printed to the screen.

  A 3D graph for performance across different color and position ranges can be generated as well:
  ```python
  colors_vs_positions(color_range_min=6, color_range_max=14, code_length_min=4, code_length_max=8, num_games=100)
  ```

### 3. Using the Mastermind GUI
You can play the Mastermind game using the GUI provided in the `GUI_Mastermind/` directory:

1. Download the `GUI_Mastermind/` directory.
2. Open `menu_page.html` in a web browser.

From there, you can choose to:
- Play the **classic game** (`manual.html`).
- **Play against the AI** (`Genetic_game.html`) using a genetic algorithm-based agent.

### Output Directory

The `output/` directory contains:
- **Plots**: Performance metrics such as learning curves and the number of guesses over time, generated during algorithm execution.
- **CSV Files**: Includes data such as algorithm performance, hyperparameter configurations, and other metrics for analysis.
