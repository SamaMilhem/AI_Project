import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv

# Configuration for Mastermind game
DEFAULT_COLORS = ['B', 'G', 'K', 'R', 'W', 'Y']  # Possible colors
DEFAULT_INITIAL_GUESS = ['B', 'B', 'G', 'G']  # initial guess

# Genetic algorithm parameters
DEFAULT_MAX_POP_SIZE = 150
DEFAULT_MAX_GENERATIONS = 100
DEFAULT_CROSSOVER_PROBABILITY = 0.5
DEFAULT_CROSSOVER_THEN_MUTATION_PROBABILITY = 0.03
DEFAULT_PERMUTATION_PROBABILITY = 0.03
DEFAULT_INVERSION_PROBABILITY = 0.02
DEFAULT_ELITE_RATIO = 0.4
DEFAULT_WHITE_PEGS_WEIGHT = 1
DEFAULT_BLACK_PEGS_WEIGHT = 1
DEFAULT_C0DE_LENGTH = 4  # Length of the code (number of pegs)
DEFAULT_NUM_COLORS = 6 # Number of possible colors in the code
random.seed(42)

def evaluate_guess(guess, secret):
    """
    Compares the AI's guess with the secret code.
    Returns the number of pegs that are correct in both color and position (black pins)
    and the number of pegs that are correct in color but wrong in position (white pins).

    Args:
        guess (list): The AI's guess for the secret code.
        secret (list): The actual secret code.

    Returns:
        tuple: (black pins, white pins)
    """
    assert len(guess) == len(secret)
    copy_secret = secret[:]
    copy_guess = guess[:]

    black_pins = 0  # Correct color and position
    white_pins = 0  # Correct color, wrong position

    # Find black pins
    for i in range(len(secret)):
        if secret[i] == guess[i]:
            black_pins += 1
            copy_secret[i] = 51
            copy_guess[i] = 5151

    # Find white pins
    for code in copy_guess:
        if code in copy_secret:
            white_pins += 1
            copy_secret[copy_secret.index(code)] = 51

    return black_pins, white_pins


def calculate_fitness(trial, previous_guesses, white_pegs_weight, black_pegs_weight):
    """
    Calculates the fitness score of a trial code.

    Args:
        trial (list): The trial code to evaluate.
        previous_guesses (list): A list of tuples [(guess, result)], where 'guess' is a previous guess, and
        'result' is a tuple (black_pegs, white_pegs).
        white_pegs_weight (int or float): Weight for white pegs (correct color, wrong position).
        black_pegs_weight (int or float): Weight for black pegs (correct color and position).

    Returns:
        int: The fitness score (lower is better).
    """

    def get_difference(trial, guess):
        guess_result = guess[1]
        guess_code = guess[0]
        trial_result = evaluate_guess(trial, guess_code)
        dif = [abs(trial_result[i] - guess_result[i]) for i in range(2)]
        return tuple(dif)

    differences = [get_difference(trial, guess) for guess in previous_guesses]
    sum_black_pin_differences = sum(dif[0] for dif in differences)
    sum_white_pin_differences = sum(dif[1] for dif in differences)

    fitness_score = black_pegs_weight * sum_black_pin_differences + white_pegs_weight * sum_white_pin_differences
    return fitness_score


def crossover(code1, code2):
    """
    Performs one-point or two-point crossover between two codes.

    Args:
        code1 (list): First parent code.
        code2 (list): Second parent code.

    Returns:
        tuple: Two offspring codes resulting from crossover.
    """
    if random.random() < 0.5:
        point = random.randint(1, len(code1) - 1)
        return code1[:point] + code2[point:], code2[:point] + code1[point:]
    else:
        point1, point2 = sorted(random.sample(range(1, len(code1)), 2))
        return (code1[:point1] + code2[point1:point2] + code1[point2:],
                code2[:point1] + code1[point1:point2] + code2[point2:])


def mutate(code, slots, colors):
    if random.random() < DEFAULT_CROSSOVER_THEN_MUTATION_PROBABILITY:
        i = random.randint(0, slots - 1)
        code[i] = random.choice(colors)
    return code


def permute(code, slots):
    if random.random() < DEFAULT_PERMUTATION_PROBABILITY:
        pos1, pos2 = random.sample(range(slots), 2)
        code[pos1], code[pos2] = code[pos2], code[pos1]
    return code


def invert(code, slots):
    if random.random() < DEFAULT_INVERSION_PROBABILITY:
        pos1, pos2 = sorted(random.sample(range(slots), 2))
        code[pos1:pos2 + 1] = reversed(code[pos1:pos2 + 1])
    return code


def evolve_population(popsize, fitness_function, eliteratio, slots, colors):
    """
    Performs a genetic algorithm to evolve a population of possible codes towards the correct code.

    Args:
        popsize (int): The size of the population (number of codes in each generation).
        fitness_function (function): A function to evaluate the fitness of each code.
        eliteratio (float): The ratio of the population to be kept as elite individuals for the next generation.
        slots (int): The number of slots in the code (length of the code).
        colors (list): A list of possible colors or values that can be chosen to form the codes.

    Returns:
        list: A list of eligible codes that could potentially be the correct code.
    """
    population = [[random.choice(colors) for _ in range(slots)] for _ in range(popsize)]

    for _ in range(DEFAULT_MAX_GENERATIONS):
        fitness_scores = [(fitness_function(ind), ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0])

        num_elite = int(eliteratio * popsize)
        elite = [ind for _, ind in fitness_scores[:num_elite]]

        new_population = elite.copy()
        while len(new_population) < popsize:
            parents = random.sample(elite, 2)

            offspring1, offspring2 = crossover(parents[0], parents[1])
            offspring1 = mutate(permute(invert(offspring1, slots), slots), slots, colors)
            offspring2 = mutate(permute(invert(offspring2, slots), slots), slots, colors)
            new_population.extend([offspring1, offspring2])

        population = new_population[:popsize]

        eligibles = [ind for score, ind in fitness_scores if score == 0]
        if eligibles:
            return eligibles

    return population[:popsize]


def simulate_games(num_simulations, colors, code_length, pop_size, white_pegs_weight, black_pegs_weight,
                   initial_guess, elite_ratio, penalty_weight=0):
    """
    Runs multiple simulations of the Mastermind game and reports the average performance.

    Args:
        num_simulations (int): The number of game simulations to run.
        colors (list): A list of possible colors or values to be used in the code.
        code_length (int): The length of the secret code (number of slots to fill).
        pop_size (int): The size of the population for the genetic algorithm in each generation.
        white_pegs_weight (int or float): The weight assigned to white pegs.
        black_pegs_weight (int or float): The weight assigned to black pegs.
        initial_guess (list): The initial guess to start the game with.
        elite_ratio (float): The ratio of the elite population to carry over between generations.
        penalty_weight(int): The penalty added to the fitness scores

    Returns:
        float: The average number of turns taken to win the game across successful simulations.
    """

    total_turns = 0
    successful_simulations = 0

    for _ in range(num_simulations):
        TOGUESS = [random.choice(colors) for _ in range(code_length)]
        code = initial_guess
        turn = 1

        previous_guesses = []

        def fitness_function(trial):
            return (calculate_fitness(trial, previous_guesses, white_pegs_weight, black_pegs_weight) +
                    penalty_weight * code_length * (turn-1))

        result = evaluate_guess(code, TOGUESS)
        previous_guesses.append((code, result))

        while result != (code_length, 0):
            eligibles = evolve_population(pop_size, fitness_function, elite_ratio, code_length, colors)

            if len(eligibles) == 0:
                code = [random.choice(colors) for _ in range(code_length)]
            else:
                code = eligibles.pop()

            while code in [c for (c, r) in previous_guesses]:
                if len(eligibles) == 0:
                    code = [random.choice(colors) for _ in range(code_length)]
                else:
                    code = eligibles.pop()

            turn += 1
            result = evaluate_guess(code, TOGUESS)
            previous_guesses.append((code, result))

            if result == (code_length, 0):
                successful_simulations += 1
                total_turns += turn
                break

    average_turns = total_turns / successful_simulations if successful_simulations > 0 else float('inf')
    return average_turns


def colors_vs_positions(color_range_min, color_range_max, code_length_min, code_length_max, num_games):
    """    The function simulates genetic algorithm games for different
    configurations of colors and positions.
    """
    colors_range = list(range(color_range_min, color_range_max+1))
    positions_range = list(range(code_length_min, code_length_max+1))

    # Create an empty DataFrame to store results for average turns and time taken
    results_turns = pd.DataFrame(columns=positions_range, index=colors_range)
    results_time = pd.DataFrame(columns=positions_range, index=colors_range)

    for color in colors_range:
        for position in positions_range:
            # Generate a list of colors from 'A' to the specified color limit
            colors_range_list = list(string.ascii_uppercase[:color])
            code = [random.choice(colors_range_list) for _ in range(position)]
            start_time = time.time()

            # Run the simulation and get the average turns
            avg_turns =   simulate_games(num_simulations=num_games, colors=colors_range_list, code_length=position,
                                           pop_size=DEFAULT_MAX_POP_SIZE, white_pegs_weight=DEFAULT_WHITE_PEGS_WEIGHT,
                                           black_pegs_weight=DEFAULT_BLACK_PEGS_WEIGHT,
                                           initial_guess=code, elite_ratio=DEFAULT_ELITE_RATIO)

            # Calculate the time taken for the simulation
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Store the result in the DataFrame
            results_turns.at[color, position] = avg_turns
            results_time.at[color, position] = elapsed_time/100

            # Print results for each color and position
            print(
                f"Colors: {color}, Positions: {position}, Avg Turns: {avg_turns:.2f}, Time:"
                f" {elapsed_time/100:.2f} seconds")

    # Display the results tables
    print("\nAverage Turns Table:")
    print(results_turns)

    print("\nTime Taken (seconds) Table:")
    print(results_time)
    plot_results(results_turns, results_time, positions_range, colors_range)


def plot_surface_helper(X, Y, Z, xlabel, ylabel, zlabel, title, cmap, save_path):
    """
    Helper function to create a 3D surface plot.

    Args:
        X (ndarray): X-axis values (positions).
        Y (ndarray): Y-axis values (colors).
        Z (ndarray): Z-axis values (result data to plot).
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        zlabel (str): Label for the Z-axis.
        title (str): Title for the plot.
        cmap (str): Colormap to use for the surface plot.
        save_path (str): Path to save the figure.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')

    # Customize ticks
    ax.set_xticks(np.arange(min(X.flatten()), max(X.flatten()) + 1, 1))
    ax.set_yticks(np.arange(min(Y.flatten()), max(Y.flatten()) + 1, 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=15)

    # Save the figure
    fig.savefig('Output/' + save_path, format='png', dpi=300)
    print(f"Figure saved as {save_path}")


def plot_results(results_turns, results_time, positions_range, colors_range,
                 save_path_turns='turns_plot_genetic_alg.png', save_path_time='time_plot_genetic_alg.png',
                 algorithm_name='The New Genetic Algorithm'):
    """
    This function plots the average number of guesses and the average time taken for various configurations
    of colors and positions in the Mastermind game. It generates 3D surface plots and saves them to the specified
     file paths.

    Args:
        results_turns (DataFrame): The DataFrame containing the average number of guesses.
        results_time (DataFrame): The DataFrame containing the average time taken.
        positions_range (list): The range of code lengths (positions).
        colors_range (list): The range of color options.
        save_path_turns (str): File path to save the plot for average guesses.
        save_path_time (str): File path to save the plot for average time.
        algorithm_name (str): The name of the algorithm used for simulation, included in the plot titles.
    """
    # Compute X and Y meshgrid once
    X, Y = np.meshgrid(positions_range, colors_range)

    # Plot Average Turns
    Z_turns = np.array(results_turns, dtype=float)
    plot_surface_helper(X, Y, Z_turns,
                        xlabel='Code Length (Positions)',
                        ylabel='Number of Colors',
                        zlabel='Average Number of Guesses',
                        title=f'Impact of Colors and Code Length on Average Guesses\n{algorithm_name}',
                        cmap='plasma',
                        save_path=save_path_turns)

    # Plot Time Taken
    Z_time = np.array(results_time, dtype=float)
    plot_surface_helper(X, Y, Z_time,
                        xlabel='Code Length (Positions)',
                        ylabel='Number of Colors',
                        zlabel='Average Time Taken (seconds)',
                        title=f'Impact of Colors and Code Length on The Average Time Taken\n{algorithm_name}',
                        cmap='plasma',
                        save_path=save_path_time)

def different_weights(max_weight_range=3, min_weight_range=1, max_penalty_weight=4, min_penalty_weight=0, num_games=100):
    """
    This function explores different combinations of weights for black pegs, white pegs, and penalty weight in
    the Genetic Algorithm applied to the game
    Args:
            max_weight_range (int): Maximum value for both black and white peg weights (default is 3).
            min_weight_range (int): Minimum value for both black and white peg weights (default is 1).
            max_penalty_weight (int): Maximum value for the penalty weight (default is 4).
            min_penalty_weight (int): Minimum value for the penalty weight (default is 0).
            num_games (int): Number of games to simulate for each combination (default is 100).

    """
    # Create a list to store the results
    results = []

    # Set to track unique ratios of black_pegs_weight to white_pegs_weight
    unique_ratios = set()

    # Variables to track the best combination (minimal guesses)
    min_guesses = float('inf')
    best_combination = None

    # Loop over different weights for black, white pegs, and penalty weights
    for i in range(min_weight_range, max_weight_range+1):
        for j in range(i, max_weight_range+1):
            # Calculate the ratio between black and white pegs weights
            ratio = i / j
            # Only proceed if this ratio hasn't been seen before
            if ratio not in unique_ratios:
                unique_ratios.add(ratio)
                # Loop over different penalty weights
                for penalty_weight in range(min_penalty_weight, max_penalty_weight+1):
                    # Simulate the games with the given parameters
                    avg_guesses = simulate_games(num_simulations=num_games, colors=DEFAULT_COLORS,
                                                 code_length=DEFAULT_C0DE_LENGTH, pop_size=DEFAULT_MAX_POP_SIZE,
                                                 white_pegs_weight=j, black_pegs_weight=i,
                                                 initial_guess=DEFAULT_INITIAL_GUESS, elite_ratio=DEFAULT_ELITE_RATIO,
                                                 penalty_weight=penalty_weight)
                    avg_guesses = round(avg_guesses, 2)
                    # Add the results to the list
                    results.append({"black_pegs_weight": i, "white_pegs_weight": j, "penalty_weight": penalty_weight,
                        "average_guesses": avg_guesses})
                    # Update the best combination if this one has fewer guesses
                    if avg_guesses < min_guesses:
                        min_guesses = avg_guesses
                        best_combination = (i, j, penalty_weight)
                    # Print the result (optional)
                    print( f"Results: For black_pegs_weight = {i}, white_pegs_weight = {j}, penalty_weight = "
                        f"{penalty_weight}, the average number of guesses is {avg_guesses}.\n")

    sorted_results = sorted(results, key=lambda x: x['average_guesses'])
    # Save the results to a CSV file
    with open('Output/fitness_weights_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["black_pegs_weight", "white_pegs_weight", "penalty_weight",
                                                  "average_guesses"])
        writer.writeheader()
        writer.writerows(sorted_results)
    # Print and save the best combination with minimal guesses
    if best_combination:
        best_black_weight, best_white_weight, best_penalty_weight = best_combination
        print(
            f"The best combination is black_pegs_weight = {best_black_weight}, white_pegs_weight = {best_white_weight},"
            f" penalty_weight = {best_penalty_weight}, with an average of {min_guesses} guesses.")

    print("Results have been saved to fitness_weights_results.csv.")


if __name__ == '__main__':
    colors_vs_positions(color_range_min=6, color_range_max=14, code_length_min=4, code_length_max=8, num_games=100)
    different_weights()