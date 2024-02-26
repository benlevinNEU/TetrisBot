import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, sys, re, copy, time
from frame_tetris import TetrisApp, COLS, ROWS
import multiprocessing
from multiprocessing import Process, Lock

from pynput import keyboard
from pynput.keyboard import Key

import cProfile

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from get_latest_profiler_data import print_stats
import numpy as np

# Use all execpt 1 of the available cores
MAX_WORKERS = multiprocessing.cpu_count() # TODO: Add -1 if you don't want to use all cores
MAX_WORKERS = 1 # TODO: Remove this line to use all cores
PROF_DIR = "./frame-tetris/profiler/"

class TetrisNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(TetrisNet, self).__init__()
        layers = []
        previous_layer_size = input_size
        for hidden_layer_size in hidden_layers:
            layer = nn.Linear(previous_layer_size, hidden_layer_size)
            layers.append(layer)
            layers.append(nn.ReLU())
            previous_layer_size = hidden_layer_size
        layers.append(nn.Linear(previous_layer_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def apply_custom_initialization(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, -0.01, 0.01)

def mutate_network(network, mutation_rate=0.01, mutation_strength=0.1):
    with torch.no_grad():
        for param in network.parameters():
            if len(param.shape) > 1:  # Weights of linear layers
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        if np.random.rand() < mutation_rate:
                            param[i][j] += mutation_strength * torch.randn(1).item()  # Use .item()
            else:  # Biases of linear layers
                for i in range(param.shape[0]):
                    if np.random.rand() < mutation_rate:
                        param[i] += mutation_strength * torch.randn(1).item()  # Use .item()


def evaluate_network(args):
    """
    Evaluate the network by playing a game using the provided game function.
    
    Args:
        network (nn.Module): The neural network to evaluate.
        game_func (callable): A function that takes an action (int) and returns the
                              flattened game board, score, and game over flag.
                              
    Returns:
        int: The final score achieved by the network.
    """

    index, network, plays, gp, lock, results_list, slot, profile = args

    if profile[0]:
        profiler = cProfile.Profile()
        profiler.enable()

    with torch.no_grad():  # Ensure no gradients are computed during evaluation
        network.eval()  # Set the network to evaluation mode

        total_score = 0

        width = gp["cell_size"] * (gp["cols"] + 6)
        height = gp["cell_size"] * gp["rows"] + 80
        pos = ((width * slot) % 2560, height * int(slot / int(2560 / width)))

        for i in range(plays):

            game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], window_pos=pos)

            # Initialize the game
            board, next_piece = game.get_state()
            game_over = False
            
            move = 0
            
            while not game_over and move < 10000: # TODO: Use Simulated anealing to reduce this number to incentivise quicker games

                mask = game.get_possible_states()

                # Flatten the game board and append the next piece
                flattened_board = np.array(board).flatten().tolist() + [next_piece]

                # Convert flattened_board to tensor and add batch dimension
                board_tensor = torch.tensor(flattened_board, dtype=torch.float32).unsqueeze(0)

                # Forward pass through the network
                output = network(board_tensor)

                masked_output = output * mask
                
                # Select the action with the highest output value
                # Adjust this if your network's output does not directly correspond to action indices
                _, predicted_action = torch.max(masked_output, 1)
                action = predicted_action.item()
                
                # Perform the action in the game
                board, next_piece, score, game_over = game.ai_command(action)

                move += 1

            # Accumulate the score over multiple games
            total_score += score

            utils.print_to_line(lock, f"Network {index} - Game {i + 1}/{plays} - Final score: {total_score}\n", index)

    if profile:
        profiler.disable()
        t = int(time.time())
        profiler.dump_stats(f"{PROF_DIR}{profile[1]}/proc{t}.prof")

    results_list.append((index, total_score))

def evaluate_population(population, plays, game_params, profile=(False, 0)):
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    results_list = manager.list()  # Managed list for collecting results

    processes = []
    for i, net in enumerate(population):
        if i < MAX_WORKERS:
            slot = i

        if len(processes) >= MAX_WORKERS:
            # Continuously check if any process has finished
            while True:
                # Check each process in the list
                for p, open_slot in processes[:]:
                    if not p.is_alive():
                        processes.remove((p, open_slot))
                        slot = open_slot

                # Break the loop if we are under the max_workers limit
                if len(processes) < MAX_WORKERS:
                    break
                # Avoid tight loop with a short sleep
                time.sleep(0.01)

        # Start a new process
        args = (i, net, plays, game_params, lock, results_list, slot, profile)
        p = multiprocessing.Process(target=evaluate_network, args=(args,))
        p.start()
        processes.append((p, slot))

    # Wait for all remaining processes to complete
    for p, _ in processes:
        p.join()

    # Convert the manager list to a regular list for further processing
    results = list(results_list)
    return results

def load_network(filepath, input_size, hidden_layers, output_size, device='cpu'):
    """
    Load a network from a file.

    Args:
        filepath (str): Path to the .pth file.
        input_size (int): Size of the input layer.
        hidden_layers (list): List of sizes of the hidden layers.
        output_size (int): Size of the output layer.
        device (str): Device to load the network onto, 'cpu' or 'cuda'.

    Returns:
        TetrisNet: The loaded network.
    """
    network = TetrisNet(input_size, hidden_layers, output_size)
    state_dict = torch.load(filepath, map_location=device)
    network.load_state_dict(state_dict)
    return network

def mutate_next_gen(top_networks, population_size, mutation_rate=0.01, mutation_strength=0.1):
        next_generation = []
        for net in top_networks:
            for _ in range(int(population_size / top_n)):
                new_net = copy.deepcopy(net)
                mutate_network(new_net, mutation_rate, mutation_strength)
                next_generation.append(new_net)

        return next_generation

if __name__ == "__main__":

    profile = True
    tid = int(time.time())

    if profile: os.makedirs(f"{PROF_DIR}{tid}")

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    multiprocessing.set_start_method('spawn')

    # Parameters for the evolutionary process
    population_size = 100
    top_n = 10
    generations = 420
    plays = 3  # Number of games to play for each network each generation

    input_size = COLS * ROWS + 1
    hidden_layers = [128, 64]  # Example hidden layers sizes
    output_size = 4*COLS  # Adjust based on the number of possible actions

    # Initialize the game
    game_params = {
        "gui": False,  # Set to True to visualize the game
        "cell_size": 10,
        "cols": COLS,
        "rows": ROWS,
        "window_pos": (0, 0)
    }

    print("Starting Evolutionary Training\n\n\n\n\n\n\n")

    networks_dir = "./frame-tetris/networks"
    M = 5  # Number of top networks to load
    top_networks_info = utils.find_top_networks(networks_dir, M)
    
    if len(top_networks_info) == 0:
        # Initialize population with custom initialization
        population = [TetrisNet(input_size, hidden_layers, output_size) for _ in range(population_size)]
        for net in population:
            net.apply(apply_custom_initialization)
    else:
        # Load the top M networks from the file
        top_networks = [utils.load_network(path, input_size, hidden_layers, output_size, device) for _, path in top_networks_info]
        population = mutate_next_gen(top_networks, population_size, mutation_rate=0.01, mutation_strength=0.1)

    exit = False

    def on_press(key):
        # Check if the pressed key is ESC
        if key == Key.esc:
            global exit
            exit = True
            return False

    # Set up the listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
        
    for generation in range(utils.get_newest_generation_number(networks_dir) + 1, generations):

        # Evaluate all networks in parallel
        results = evaluate_population(population, plays, game_params, (profile, tid))

        # Convert results to a list and sort by the score
        results = sorted(results, key=lambda x: x[1], reverse=True)  # Now sorting the list of results

        # Select top performers
        top_performers = results[:top_n]

        utils.save_networks(networks_dir, [(population[index], score) for index, score in top_performers], generation)

        # Print generation info
        sys.stdout.write(f'\033[{population_size+1}B')
        sys.stdout.flush()
        print(f"Generation {generation}: Top Score = {top_performers[0][1]}\n\n\n\n\n")

        # Extract the networks of the top performers
        top_networks = [population[index] for index, _ in top_performers]

        population = mutate_next_gen(top_networks, population_size, mutation_rate=0.05, mutation_strength=0.2)

        # Listen for escape key and break the loop if pressed
        if exit:
            break

    print("Finished Evolutionary Training")

    if profile:
        profiler.disable()
        profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

        profiler_dir = f"{PROF_DIR}{tid}"
        directory = './frame-tetris'

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)
