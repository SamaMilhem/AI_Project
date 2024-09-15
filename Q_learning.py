import logging
import random
import re
import time
from collections import Counter
import itertools
import pandas as pd
import os
import concurrent.futures
from matplotlib import pyplot as plt
import seaborn as sns
import genetic_algorithm

# Total possible guesses (6^4 for a 4-peg code with 6 colors)
MAX_GUESSES = genetic_algorithm.DEFAULT_NUM_COLORS ** genetic_algorithm.DEFAULT_C0DE_LENGTH
EPSILON = 0.4  # Exploration probability for epsilon-greedy strategy
ALPHA = 0.3  # Learning rate for Q-learning
DISCOUNT = 0.8  # Discount factor for future rewards
MAX_EPISODES = 2000   # Number of training episodes
GUESSES_THRESHOLD = 50
DEFAULT_BATCH_SIZE = 5
random.seed(42)

#
# # Setup logger
# logging.basicConfig(filename='Output/mastermind_q_learning.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
#
# logging.info("Initializing Q-learning with parameters: Alpha = %s, Discount = %s, "
#              "Epsilon = %s", ALPHA, DISCOUNT, EPSILON)

class Environment:
    """
    Environment class for the Mastermind game.
    This class handles the game's logic, including converting between indices and guesses, scoring feedback,
     and determining rewards.
    """

    def __init__(self, secret, code_length, num_colors):
        """
        Initialize the environment with a secret code.
        Args:
            secret (int or str): The secret code as a string or an index.
        """
        self.secret = self._number_from_index(secret, num_colors,code_length) if isinstance(secret, int) else secret

    @staticmethod
    def _index_from_number(number, num_colors):
        """
        Convert a 4-digit guess to an index.
        Args:
            number (str): A 4-digit string representing the guess.
        Returns:
            int: The corresponding index.
        """
        return int(number, base=num_colors)

    @staticmethod
    def _number_from_index(index, num_colors, code_length):
        """
        Convert an index to a 4-digit guess.
        Args:
            index (int): The index to convert.
        Returns:
            str: The corresponding 4-digit guess.
        """
        digits = []
        while index > 0:
            digits.append(str(index % num_colors))
            index //= num_colors
        return "".join(reversed(digits)).zfill(code_length)

    @staticmethod
    def score(p, q):
        """
        Calculate the feedback (hits and misses) for a guess compared to the secret code.
        Args:
            p (str): The secret code.
            q (str): The guess.
        Returns:
            tuple: A tuple containing the number of hits and misses.
        """
        hits = sum(p_i == q_i for p_i, q_i in zip(p, q))
        misses = sum((Counter(p) & Counter(q)).values()) - hits
        return hits, misses

    def get_feedback(self, action):
        """
        Get feedback for the current guess against the secret code.
        Args:
            action (str): The guess made by the agent.
        Returns:
            tuple: The feedback as (hits, misses).
        """
        return self.score(self.secret, action)

    def reward(self, guess):
        """
        Determine the reward for a guess.
        Args:
            guess (str): The guess made by the agent.
        Returns:
            int: 1 if the guess is correct, -1 otherwise.
        """
        return 1 if guess == self.secret else -1

class QLearningAgent:
    """
    Q-learning agent for the Mastermind game.
    This agent learns the best strategy to guess the secret code through reinforcement learning.
    """
    def __init__(self, epsilon, alpha, discount, max_guesses, num_colors, code_length):
        """
        Initialize the Q-learning agent with given parameters.
        Args:
            epsilon (float): The exploration probability for epsilon-greedy strategy.
            alpha (float): The learning rate.
            discount (float): The discount factor for future rewards.
        """
        self.possible_states = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.colors = num_colors
        self.code_length = code_length
        self.Q_values = Counter()  # Dictionary to store Q-values for state-action pairs
        self.reset_possible_states(max_guesses)

    def reset_possible_states(self, max_guesses):
        """
        Reset the possible states (guesses) for the agent.
        This is used to start each episode with a full set of possible guesses.
        """
        self.possible_states = [Environment._number_from_index(idx, self.colors, self.code_length)
                                for idx in range(max_guesses)]

    def restrict_possible_states(self, guess, feedback):
        """
        Restrict the possible states based on the feedback from a guess.
        This method filters out states that would not produce the same feedback as the given guess.
        Args:
            guess (str): The guess made by the agent.
            feedback (tuple): The feedback received for the guess.
        """
        self.possible_states = [state for state in self.possible_states if Environment.score(guess, state) == feedback]

    def select_move(self):
        """
        Select the next move using an epsilon-greedy strategy.
        The agent may either explore (choose a random action) or exploit (choose the best-known action).
        Returns:
            str: The selected move (guess).
        """
        best_move = self.get_best_action()
        return best_move if random.random() > self.epsilon else self.random_action()

    def get_best_action(self):
        """
        Select the best action based on the current Q-values.
        If multiple actions have the same Q-value, one is chosen randomly.
        Returns:
            str: The best action (guess).
        """
        max_value = max(self.Q_values[state] for state in self.possible_states)
        best_actions = [state for state in self.possible_states if self.Q_values[state] == max_value]
        return random.choice(best_actions)
    def random_action(self):
        """
        Select a random action from the possible states.
        Returns:
            str: A randomly chosen action (guess).
        """
        return random.choice(self.possible_states)

    def update(self, state, reward):
        """
        Update the Q-values using the Q-learning update rule.
        Args:
            state (str): The current state (guess).
            reward (int): The reward received for the action.
        """
        best_next_value = max(self.Q_values[next_state] for next_state in self.possible_states)
        td_target = reward + self.discount * best_next_value
        self.Q_values[state] += self.alpha * (td_target - self.Q_values[state])

    def learn(self, state, action, feedback, reward):
        """
        Learn from the action taken and its feedback by updating the Q-values.
        Args:
            state (str): The current state (guess).
            action (str): The action taken (guess).
            feedback (tuple): The feedback received for the guess.
            reward (int): The reward received for the action.
        """
        self.restrict_possible_states(action, feedback)
        self.update(state, reward)

def train(agent, n_episodes, max_guesses, num_colors, code_length):
    """
        Train the Q-learning agent over multiple episodes.

        Args:
            agent (QLearningAgent): The agent to be trained.
            n_episodes (int): The number of training episodes.
            max_guesses (int): The maximum number of guesses allowed per game.
            num_colors (int): The number of possible colors in the game (Mastermind).
            code_length (int): The length of the secret code (number of positions in the code).
    """

    for _ in range(n_episodes):
        secret = random.randint(0, max_guesses - 1)  # Generate a random secret code
        env = Environment(secret,code_length, num_colors)
        agent.reset_possible_states(max_guesses)
        guess = agent.random_action()  # Start with a random guess

        while guess != env.secret:
            feedback = env.get_feedback(guess)
            reward = env.reward(guess)
            agent.learn(guess, guess, feedback, reward)
            if reward == 1:  # Correct guess, end episode
                break
            guess = agent.select_move()  # Select the next guess

def evaluate_agent(agent, secret, code_length, max_guesses, num_colors):
    """
    Evaluate the Q-learning agent by determining the number of guesses it needs to find the secret code.

    Args:
        agent (QLearningAgent): The Q-learning agent to be evaluated.
        secret (str or int): The secret code that the agent is trying to guess.
        code_length (int): The length of the secret code (number of positions in the code).
        max_guesses (int): The maximum number of guesses allowed in the game.
        num_colors (int): The number of possible colors in the game (Mastermind).

    Returns:
        int: The number of guesses needed by the agent to find the secret code.
        The agent stops after GUESSES_THRESHOLD guesses if it doesn't find the secret - to prevent infinite loops.
    """

    agent.reset_possible_states(max_guesses)
    guess = agent.get_best_action()
    env = Environment(secret, code_length, num_colors)
    num_guesses = 1

    while guess != env.secret and num_guesses < GUESSES_THRESHOLD:
        feedback = env.get_feedback(guess)
        agent.restrict_possible_states(guess, feedback)
        guess = agent.get_best_action()
        num_guesses += 1

    return num_guesses

def generate_hyperparameter_table(num_of_games, num_colors=genetic_algorithm.DEFAULT_NUM_COLORS,
                                  code_length=genetic_algorithm.DEFAULT_C0DE_LENGTH):
    """
    Generate a table of Alpha, Discount, Epsilon combinations with their Average Number of Guesses.
    Then find and display the combination with the minimal average number of guesses.
    """
    alphas = [0.1, 0.2, 0.3, 0.4]  # Learning rate values
    discounts = [0.5, 0.7, 0.8, 0.9]  # Discount factor values
    epsilons = [0.1, 0.2, 0.3, 0.4]  # Epsilon values for exploration

    # List to store results
    results = []

    # Iterate over all possible combinations of alpha, discount, and epsilon
    for alpha, discount, epsilon in itertools.product(alphas, discounts, epsilons):
        print(f"Running simulation with alpha={alpha}, discount={discount}, epsilon={epsilon}")

        # Run the simulation for the current combination
        avg_guesses, _,_ = simulate_games(
            epsilon=epsilon,
            num_games=num_of_games,
            alpha=alpha,
            discount=discount,
            max_guesses=MAX_GUESSES,
            code_length=code_length,
            num_colors=num_colors
        )

        # Append the tuple (alpha, discount, epsilon, avg_guesses) to results
        results.append((alpha, discount, epsilon, avg_guesses))

    # Convert the results into a pandas DataFrame for easy display
    df = pd.DataFrame(results, columns=['Alpha', 'Discount', 'Epsilon', 'Avg Guesses'])
    df= df.sort_values(by="Avg Guesses").reset_index(drop=True)
    # Display the DataFrame
    print("\nTable of Hyperparameter Combinations and their Avg Guesses:")
    print(df)

    # Find the combination with the minimal average number of guesses
    min_row = df.loc[df['Avg Guesses'].idxmin()]
    print("\nCombination with the minimal average number of guesses:")
    print(min_row)

    # Optionally, save the DataFrame to a CSV file if needed
    df.to_csv('Output/hyperparameter_results.csv', index=False)

    return df, min_row


def simulate_games(epsilon, num_games, alpha, discount, max_guesses, code_length, num_colors):
    """
    Simulate multiple games, including the time for training the Q-learning agent, and evaluate performance.
    If a game fails (i.e., takes more than 50 guesses), it will not be included in the average time and guess calculation.
    """
    # Track the total time including training
    total_training_time = time.time()

    # Train the Q-learning agent
    agent = QLearningAgent(epsilon, alpha, discount, max_guesses, num_colors, code_length)
    print(f'Started the agent training, color: {num_colors}, code_length:{code_length}')
    train(agent, MAX_EPISODES, max_guesses, num_colors, code_length)
    print(f'Finished the agent training, color: {num_colors}, code_length:{code_length}')

    total_training_time = time.time() - total_training_time

    # print(f'Total training time, colors size: {num_colors} code length: {code_length}', total_training_time)

    # Variables to track total successful games, guesses, and time
    total_guesses = 0
    total_game_time = 0
    successful_games = 0

    # Simulate each game
    while successful_games < num_games:
        start_time = time.time()  # Start the timer for the current game
        secret = random.randint(0, max_guesses - 1)  # Generate a random secret code
        guesses = evaluate_agent(agent, secret, code_length, max_guesses, num_colors)

        # If the game fails (more than 50 guesses), skip and restart the game
        if guesses == 50:
            # print("Game Failed - restart game and timer")
            continue  # Do not count this game and retry

        # If successful, calculate the time for the game and accumulate the results
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_game_time += elapsed_time
        total_guesses += guesses
        successful_games += 1

        # Display game details
        # print(f'Game {successful_games}: Secret = {Environment._number_from_index(secret, num_colors, code_length)}, '
        #       f'Guesses = {guesses}, Time: {elapsed_time:.2f} seconds')

    # Calculate the averages for guesses and time, excluding failed games
    average_guesses = total_guesses / num_games

    # Include the total training time for the final output
    average_game_time = total_game_time / num_games

    # Display final results
    # print(f"\nSimulated {num_games} successful games.")
    # print(f"Average number of guesses per game: {average_guesses:.2f}")
    # print(f"Average time per game (including training): {average_game_time:.2f} seconds")

    return average_guesses, average_game_time, total_training_time


import concurrent.futures
def simulate_batch_of_games(color, position, num_of_games, epsilon, alpha, discount, batch_size=10):
    avg_turns, avg_time, ttl_training= 0, 0, 0
    for _ in range(batch_size):
        turns, time, total_training = simulate_games(epsilon, num_of_games, alpha, discount,
                                            max_guesses=color ** position, code_length=position, num_colors=color)
        avg_turns += turns
        avg_time += time
        ttl_training += total_training
    return color, position, avg_turns / batch_size, avg_time / batch_size, ttl_training/batch_size

def colors_vs_positions_multithreaded(num_of_games, min_range_color, max_range_color, min_range_position,
                                      max_range_position):
    """
    Simulates multiple Q-learning games over a range of color and position configurations to evaluate the
    algorithm's performance using multithreading.

    Parameters:
    -----------
    - num_of_games (int): Number of games to simulate for each configuration.
    - min_range_color (int): Minimum number of colors for the simulation.
    - max_range_color (int): Maximum number of colors for the simulation.
    - min_range_position (int): Minimum number of positions for the simulation.
    - max_range_position (int): Maximum number of positions for the simulation.

    Returns:
    --------
    - results_time (DataFrame): A DataFrame where each row corresponds to the number of colors,
    each column corresponds to the number of positions, and the values represent the average time taken
    (in seconds) for each configuration.
    - results_turns (DataFrame): A DataFrame where each row corresponds to the number of colors, each column
    corresponds to the number of positions, and the values represent the average number of turns (guesses)
    for each configuration.
    """

    # Define ranges for colors and positions
    colors_range = list(range(min_range_color, max_range_color + 1))
    positions_range = list(range(min_range_position, max_range_position + 1))

    # Create empty DataFrames to store results for average turns and time taken
    results_turns = pd.DataFrame(columns=positions_range, index=colors_range)
    results_time = pd.DataFrame(columns=positions_range, index=colors_range)

    # Inside the ThreadPoolExecutor block:
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for color in colors_range:
            for position in positions_range:
                futures.append(
                    executor.submit(simulate_batch_of_games, color, position, num_of_games, EPSILON, ALPHA, DISCOUNT,
                                    batch_size=DEFAULT_BATCH_SIZE))

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            color, position, avg_turns, avg_time, avg_training = future.result()

            # Store the results in the DataFrame
            results_turns.at[color, position] = avg_turns
            results_time.at[color, position] = avg_time

            # Print the results for each color and position
            print(f"Colors: {color}, Positions: {position}, Avg Turns: {avg_turns:.2f}, Time: {avg_time:.2f} seconds,"
                  f" Training: {avg_training:.2f}")
            logging.info(f"Colors: {color}, Positions: {position}, Avg Turns: {avg_turns:.2f}, Time: {avg_time:.2f},"
                         f" Training: {avg_training:.2f} seconds")
    # Display the results tables
    print("\nAverage Turns Table:")
    print(results_turns)

    print("\nTime Taken (seconds) Table:")
    print(results_time)

    return results_time, results_turns


def generate_plots(log_file='Output/mastermind_q_learning.log'):
    # Step 1: Extract data using regex
    with open(log_file, 'r') as file:
        log_data = file.read()

    # Define regex pattern to capture relevant data
    log_pattern = re.compile(
        r"Colors: (\d+), Positions: (\d+), Avg Turns: ([\d.]+), Time: ([\d.]+), Training: ([\d.]+) seconds")
    log_matches = log_pattern.findall(log_data)

    # Step 2: Create a DataFrame from the extracted data
    columns = ['Colors', 'Positions', 'Avg Turns', 'Avg Time', 'Training']
    df = pd.DataFrame(log_matches, columns=columns)

    # Convert numeric columns to appropriate types
    df['Colors'] = df['Colors'].astype(int)
    df['Positions'] = df['Positions'].astype(int)
    df['Avg Turns'] = df['Avg Turns'].astype(float)
    df['Avg Time'] = df['Avg Time'].astype(float)
    df['Training'] = df['Training'].astype(float)

    # Save the table as a CSV
    df.to_csv('Output/Q_Learning_Results.csv', index=False)
    print("Data saved as 'Q_Learning_Results.csv'")

    # Step 3: Generate the heatmaps
    def generate_heatmap(data, values, title, xlabel, ylabel, to_save_path ,cmap='plasma', value_suffix=''):
        pivot_data = data.pivot('Colors', 'Positions', values)
        plt.figure(figsize=(8, 6))

        # Annotate with both value and suffix (e.g., seconds)
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap=cmap,
                    cbar_kws={'label': value_suffix})

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(to_save_path, dpi=300)
        plt.show()


    # Heatmap 1: Color vs Position, Avg Turns with 'viridis' color map
    generate_heatmap(df, 'Avg Turns', 'Average Turns per Code Configuration Heatmap\n The Q-Leaning '
                                      'Algorithm', 'Code Length', 'Colors',
                     to_save_path='Output/Average_Turns_Q_Learning',cmap='Greens')

    # Heatmap 2: Color vs Position, Avg Time with 'coolwarm' color map
    generate_heatmap(df, 'Avg Time', 'Average Time Heatmap per Code Configuration'
                                     ' Heatmap\n The Q-Leaning Algorithm', 'Code Length', 'Colors',
                     to_save_path='Output/Average_Time_Q_Learning', cmap='Blues', value_suffix='Seconds')

    # Heatmap 3: Color vs Position, Training Time with 'magma' color map
    generate_heatmap(df, 'Training', 'Training Time per Code Configuration Heatmap\n The Q-Leaning Algorithm'
                     , 'Code Length', 'Colors',  to_save_path='Output/Average_Training_Q_Learning',
                     cmap='Reds', value_suffix='Seconds')



if __name__ == '__main__':

    # Simulate and evaluate the agent's performance over 1000 games - to get the 6,4 configuration performance
    # avg_turns,_ =  simulate_games(epsilon=EPSILON, num_games=1000, alpha=ALPHA, discount=DISCOUNT,
    #                                     max_guesses=MAX_GUESSES, code_length=CODE_LENGTH, num_colors=NUM_COLORS)
    # # hyperparameters tuning
    # hyperparameter_df, best_combination = generate_hyperparameter_table(code_length=4, num_colors=6,
    # num_of_games=1000)

    # get colors v.s. positions graphs
    # # results_time, results_turns = colors_vs_positions_multithreaded(num_of_games=50, min_range_color=6,
    #                                                                 max_range_color=9, min_range_position=4,
    #                                                                 max_range_position=6)
    generate_plots(log_file='Output/mastermind_q_learning.log')


