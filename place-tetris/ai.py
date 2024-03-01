import numpy as np
import os, sys, time, cProfile, multiprocessing, platform, re
from place_tetris import TetrisApp, COLS, ROWS

import utils
from get_latest_profiler_data import print_stats
from evals import *

pltfm = None
if platform.system() == 'Linux' and 'microsoft-standard-WSL2' in platform.release():
    pltfm = 'WSL'
    import curses
    #import keyboard
else:
    pltfm = 'Mac'
    from pynput import keyboard
    from pynput.keyboard import Key

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from local_params import gp, tp
# Initialize the game and training parameters in "local_params.py" (Use the following as example)
# **DO NOT COMMIT YOUR PARAMS TO GIT**
'''
import numpy as np
gp = {
    "gui": False,  # Set to True to visualize the game
    "cell_size": 20,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 0.01
}

# Initialize the training parameters
tp = {
    "population_size": 60,
    "top_n": 5,
    "generations": 1000,
    "plays": 5,
    "mutation_rate": lambda gen: 0.8 * np.exp(-0.001 * gen) + 0.1,
    "mutation_strength": lambda gen: 5 * np.exp(-0.001 * gen) + 0.1,
    "profile": True,
    "workers": 0, # Use all available cores
    "feature_transform": lambda x: np.array([x**2, x, 1]),
}
'''

MAX_WORKERS = tp["workers"] if tp["workers"] > 0 else multiprocessing.cpu_count()

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

class Model():
    def __init__(self, tp=tp, weights=[]):
        if len(weights) == 0:
            weights = np.ones(tp["feature_transform"](0).shape[0]*NUM_EVALS)
        self.weights = weights.reshape(tp["feature_transform"](0).shape[0], NUM_EVALS)

    def play(self, gp, pos):
        self.game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos)
        
        if gp["gui"]:
            self.game.update_board()

        options = self.game.getFinalStates()
        gameover = False
        score = 0

        while not gameover and len(options) > 0:
            min_cost = np.inf
            best_option = None

            for option in options:

                if option is None:
                    raise ValueError("Option is None")

                c = self.cost(option)
                if c < min_cost:
                    min_cost = c
                    best_option = option

            options, game_over, score = self.game.ai_command(best_option)

        self.game.quit_game()
        return score

    def cost(self, state):
        vals = np.array(getEvals(state))
        X = np.array([tp["feature_transform"](x) for x in vals])
        return np.sum(np.dot(X, self.weights))

    def mutate(self, args):

        index, models, gen, profile = args

        if profile[0]:
            profiler = cProfile.Profile()
            profiler.enable()
        
        children = []
        nchildren = int(tp["population_size"] / tp["top_n"])

        for _ in range(nchildren):
            new_weights = self.weights.copy()

            for i in range(len(new_weights)):
                if np.random.rand() < tp["mutation_rate"](gen):
                    new_weights[i] += tp["mutation_strength"](gen) * (np.random.randn()*2 - 1) # Can increase or decrease weights

            children.append(Model(tp, new_weights))

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

        results_list.append((self.weights.flatten(), np.mean(scores)))

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

def mutate_next_gen(top_models_weights, tp, profile=(False, 0), gen=0):
    next_generation = []

    manager = multiprocessing.Manager()
    models = manager.list()  # Managed list for collecting results

    processes = []

    count = 0
    for weights in top_models_weights:
        model = Model(tp, weights)
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

        if MAX_WORKERS > 1:
            args = (count, models, gen, profile)
            p = multiprocessing.Process(target=model.mutate, args=(args,))
            p.start()
            processes.append(p)
        else:
            model.mutate((count, models, gen, (False, 0)))

        # Simple progress bar
        progress = int((count) / tp["population_size"] * 100)
        sys.stdout.write(f"\rMutating: {progress}%")
        sys.stdout.flush()

    # Wait for all remaining processes to complete
    for p in processes:
        p.join()

    sys.stdout.write(f"\r")
    sys.stdout.flush()

    next_generation = np.array(list(models)).flatten().tolist()
    return next_generation

def main(stdscr):

    ## General Setup
    profile = tp["profile"]
    tid = int(time.time())

    if profile: os.makedirs(f"{PROF_DIR}{tid}")

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    multiprocessing.set_start_method('spawn')

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


    print("Starting Evolutionary Training")


    ## Load or Initialize Population
    nft = tp["feature_transform"](0).shape[0] # Number of fe transforms
    file_name = f"models_{gp['rows']}x{gp['cols']}_{nft}.npy"
    print(f"Saving data to: {MODELS_DIR}{file_name}")

    def prev_data_exists(model_dir):
        for fn in os.listdir(model_dir):
            if fn == file_name:
                return True
        return False

    exists = prev_data_exists(MODELS_DIR)

    if not exists:
        # Initialize population with custom initialization
        init_tp = tp.copy()
        init_tp["top_n"] = 1
        population = Model(tp=init_tp).mutate((0, [], 0, (False, tid)))
        models_info = None
        latest_generation = 0
        
    else:
        # Load the latest generation as an array
        models_data_file = os.path.join(MODELS_DIR, file_name)
        models_info = np.load(models_data_file)
        latest_generation = int(np.max(models_info[:, 1]))

    #for generation in range(utils.get_newest_generation_number(networks_dir) + 1, generations):
    for generation in range(latest_generation + 1, tp["generations"]):

        if models_info is None:
            # Evaluate all networks in parallel
            results = evaluate_population(population, tp["plays"], gp, (profile, tid))

            # Unpack the results and create a numpy array with score in the first column and weights after it
            models_info = np.array([(score,generation) + tuple(weights) for weights, score in results])

            # Select top performers
            models_info = models_info[models_info[:, 0].argsort()[::-1]]
            
        top_models_weights = models_info[:tp["top_n"], 2:]

        population = mutate_next_gen(top_models_weights, tp, profile=(profile, tid), gen=generation)

        # Evaluate all networks in parallel
        results = evaluate_population(population, tp["plays"], gp, (profile, tid))

        # Unpack the results and create a numpy array with score in the first column and weights after it
        models_info = np.array([(score,generation) + tuple(weights) for weights, score in results])

        # Select top performers
        top_performers = models_info[models_info[:, 0].argsort()[::-1]]

        # Print generation info
        print(f"Generation {generation}: Top Score = {top_performers[0,0]}")


        # Save the numpy array to the file
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        save_file = os.path.join(MODELS_DIR, file_name)

        if exists:
            models_info = np.vstack([models_info, np.load(save_file)])
        models_info = models_info[models_info[:, 0].argsort()[::-1]]

        np.save(save_file, models_info)
        

        # Check for safe exit
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


    ## Clean Up
    if profile:
        profiler.disable()
        profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

        profiler_dir = f"{PROF_DIR}{tid}"

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)

if __name__ == "__main__":
    
    if pltfm == 'WSL':
        curses.wrapper(main)
    else:
        main(None)