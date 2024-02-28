import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, sys, re, copy, time
from place_tetris import TetrisApp, COLS, ROWS
import multiprocessing
from multiprocessing import Process, Lock

import platform

pltfm = None
if platform.system() == 'Linux' and 'microsoft-standard-WSL2' in platform.release():
    pltfm = 'WSL'
    import curses
    #import keyboard
else:
    pltfm = 'Mac'
    from pynput import keyboard
    from pynput.keyboard import Key
    
import cProfile

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from get_latest_profiler_data import print_stats

from evals import *

# Use all execpt 1 of the available cores
MAX_WORKERS = multiprocessing.cpu_count() #- 2 # TODO: Add -1 if you don't want to use all cores
MAX_WORKERS = 1 # TODO: Remove this line to use all cores
PROF_DIR = "./place-tetris/profiler/"

class Model():
    #def __init__(self, weights=np.random.rand(6)*2-1):
    def __init__(self, weights=np.array([0.1, 0.4, 0.01, 0.2, 0.1, -0.2])):
        self.weights = weights

    def play(self, gp, pos):
        self.game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], window_pos=pos)
        
        if gp["gui"]:
            self.game.update_board()

        options = self.game.getFinalStates()
        gameover = False
        score = 0

        while not gameover and len(options) > 0:
            min_cost = np.inf
            best_option = None

            for option in options:
                c = self.cost(option)
                if c < min_cost:
                    min_cost = c
                    best_option = option

            options, game_over, score = self.game.ai_command(best_option)

        self.game.quit_game()
        return score

    def cost(self, state):

        vals = getEvals(state)
        return np.dot(self.weights, np.array(vals))

    def mutate(self, args):

        index, mutation_rate, mutation_strength, models, nchildren, profile = args

        if profile[0]:
            profiler = cProfile.Profile()
            profiler.enable()
        
        children = []

        for _ in range(nchildren):
            new_weights = self.weights.copy()

            for i in range(len(new_weights)):
                if np.random.rand() < mutation_rate:
                    new_weights[i] += mutation_strength * np.random.randn()

            children.append(Model(new_weights))

        if profile[0]:
            profiler.disable()
            t = int(time.time())
            profiler.dump_stats(f"{PROF_DIR}{profile[1]}/proc{index}{t}.prof")

        models.append(children)

        return children

    def evaluate(self, args):
        index, plays, gp, results_list, slot, profile = args

        if profile[0]:
            profiler = cProfile.Profile()
            profiler.enable()

        width = gp["cell_size"] * (gp["cols"] + 6)
        height = gp["cell_size"] * gp["rows"] + 80
        pos = ((width * slot) % 2560, height * int(slot / int(2560 / width)))

        scores = []
        for _ in range(plays):
            scores.append(self.play(gp, pos))

        results_list.append((index, np.mean(scores)))

        if profile[0]:
            profiler.disable()
            t = int(time.time())
            profiler.dump_stats(f"{PROF_DIR}{profile[1]}/proc{index}{t}.prof")

def evaluate_population(population, plays, game_params, profile=(False, 0)):
    manager = multiprocessing.Manager()
    results_list = manager.list()  # Managed list for collecting results

    processes = []
    for i, model in enumerate(population):
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
                time.sleep(0.001)

        if MAX_WORKERS > 1:
            # Start a new process
            args = (i, plays, game_params, results_list, slot, profile)
            p = multiprocessing.Process(target=model.evaluate, args=(args,))
            p.start()
            processes.append((p, slot))
        else: # If only one worker is available, run the function in the main process
            model.evaluate((i, plays, game_params, results_list, 0, (False, 0)))

        # Simple progress bar
        total_population = len(population)
        progress = int((i + 1) / total_population * 100)
        sys.stdout.write(f"\rGeneration Running: {progress}%")
        sys.stdout.flush()
        
    # Wait for all remaining processes to complete
    for p, _ in processes:
        p.join()

    sys.stdout.write(f"\r")
    sys.stdout.flush()

    # Convert the manager list to a regular list for further processing
    results = list(results_list)
    return results

def mutate_next_gen(top_models, population_size, mutation_rate=0.01, mutation_strength=0.1, profile=(False, 0)):
    next_generation = []

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    models = manager.list()  # Managed list for collecting results

    processes = []

    nchildren = int(population_size / len(top_models))

    count = 0
    for model in top_models:
        for _ in range(nchildren):
            count += 1

            if len(processes) >= MAX_WORKERS:
                # Continuously check if any process has finished
                while True:
                    # Check each process in the list
                    for p in processes:
                        if not p.is_alive():
                            processes.remove(p)

                    # Break the loop if we are under the max_workers limit
                    if len(processes) < MAX_WORKERS:
                        break
                    # Avoid tight loop with a short sleep
                    time.sleep(0.001)

            new_net = copy.deepcopy(model)
            if MAX_WORKERS > 1:
                args = (count, new_net, mutation_rate, mutation_strength, models, nchildren, profile)
                p = multiprocessing.Process(target=model.mutate, args=(args,))
                p.start()
                processes.append(p)
            else:
                model.mutate((count, mutation_rate, mutation_strength, models, nchildren (False, 0)))

            # Simple progress bar
            progress = int((count) / population_size * 100)
            sys.stdout.write(f"\rMutating: {progress}%")
            sys.stdout.flush()

    # Wait for all remaining processes to complete
    for p in processes:
        p.join()

    sys.stdout.write(f"\r")
    sys.stdout.flush()

    next_generation = list(models)
    return next_generation

def main(stdscr):

    profile = True
    tid = int(time.time())

    if profile: os.makedirs(f"{PROF_DIR}{tid}")

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    multiprocessing.set_start_method('spawn')

    # Parameters for the evolutionary process
    population_size = 50
    top_n = 8
    generations = 10000
    plays = 3  # Number of games to play for each network each generation

    # Initialize the game
    game_params = {
        "gui": False,  # Set to True to visualize the game
        "cell_size": 20,
        "cols": COLS,
        "rows": ROWS,
        "window_pos": (0, 0)
    }

    print("Starting Evolutionary Training")
    #print("\n\n\n\n\n\n\n")

    networks_dir = "./place-tetris/networks/"
    top_networks_info = utils.find_top_networks(networks_dir, top_n)
    population = [Model() for _ in range(population_size)] 
    '''if len(top_networks_info) == 0:
        # Initialize population with custom initialization
        population = [Model() for _ in range(population_size)]
    else:
        # Load the top M networks from the file
        top_networks = [utils.load_(path, weights) for _, path in top_networks_info]
        population = mutate_next_gen(top_networks, population_size, mutation_rate=0.2, mutation_strength=0.1)'''

    exit = False

    def on_press(key):
        global exit
        # Check if the pressed key is ESC
        if key == Key.esc:
            exit = True
            return False

    # Set up the listener
    if pltfm == 'Mac':
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    else:
        # Initialize curses environment
        curses.cbreak()  # Disable line buffering
        stdscr.keypad(True)  # Enable special keys to be recorded
        curses.noecho()  # Prevent input from being echoed to the screen
        stdscr.nodelay(True)  # Make getch() non-blocking
        
    #for generation in range(utils.get_newest_generation_number(networks_dir) + 1, generations):
    for generation in range(0 + 1, generations):

        # Evaluate all networks in parallel
        results = evaluate_population(population, plays, game_params, (profile, tid))

        # Convert results to a list and sort by the score
        results = sorted(results, key=lambda x: x[1], reverse=True)  # Now sorting the list of results

        # Select top performers
        top_performers = results[:top_n]

        #utils.save_networks(networks_dir, [(population[index], score) for index, score in top_performers], generation)

        # Print generation info
        #sys.stdout.write(f'\033[{population_size+1}B')
        #sys.stdout.flush()
        print(f"Generation {generation}: Top Score = {top_performers[0][1]}")
        #print("\n\n\n\n\n")

        # Extract the networks of the top performers
        top_networks = [population[index] for index, _ in top_performers]

        population = mutate_next_gen(top_networks, population_size, mutation_rate=0.1, mutation_strength=0.01, profile=(profile, tid))

        if pltfm == 'WSL':
            # Non-blocking check for input
            key = stdscr.getch()
            if key == 27:  # ESC key
                break
            elif key != -1:
                stdscr.refresh()

        # Listen for escape key and break the loop if pressed
        if exit:
            break

    print("Finished Evolutionary Training")

    if profile:
        profiler.disable()
        profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

        profiler_dir = f"{PROF_DIR}{tid}"
        directory = './place-tetris'

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)

if __name__ == "__main__":
    
    if pltfm == 'WSL':
        curses.wrapper(main)
    else:
        main(None)