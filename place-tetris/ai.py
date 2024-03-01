import numpy as np
import os, sys, time, cProfile, multiprocessing, platform
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

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


# Initialize the game parameters
gp = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 20,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 1
}

# Initialize the training parameters
tp = {
    "population_size": 60,
    "top_n": 5,
    "generations": 1000,
    "plays": 5,
    "mutation_rate": 0.8,
    "mutation_strength": 5,
    "profile": True,
}

# Use all execpt 1 of the available cores
MAX_WORKERS = multiprocessing.cpu_count() #- 2 # TODO: Add -1 if you don't want to use all cores
MAX_WORKERS = 1 # TODO: Remove this line to use all cores
PROF_DIR = os.path.join(current_dir, "profiler/")
MODELS_DIR = os.path.join(current_dir, "models/")

class Model():
    # TODO: Add support for weights for transformed data and biases
    # TODO: Add support for random initialization of that cover input space better
    def __init__(self, weights=np.array([0.1, 0.4, 0.01, 0.2, 0.1, -0.2])):
        self.weights = weights

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
                    new_weights[i] += mutation_strength * (np.random.randn()*2 - 1) # Can increase or decrease weights

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

        results_list.append((self.weights, np.mean(scores)))

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

def mutate_next_gen(top_models_weights, tp, profile=(False, 0)):
    next_generation = []

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    models = manager.list()  # Managed list for collecting results

    processes = []

    nchildren = int(tp["population_size"] / top_models_weights.shape[0])

    count = 0
    for weights in top_models_weights:
        model = Model(weights)
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
            args = (count, tp["mutation_rate"], tp["mutation_strength"], models, nchildren, profile)
            p = multiprocessing.Process(target=model.mutate, args=(args,))
            p.start()
            processes.append(p)
        else:
            model.mutate((count, tp["mutation_rate"], tp["mutation_strength"], models, nchildren, (False, 0)))

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
    population = [Model() for _ in range(tp["population_size"])]
    file_name = f"models_{gp['rows']}x{gp['cols']}.npy"
    print(f"Saving data to: {MODELS_DIR}{file_name}")

    def prev_data_exists(model_dir):
        for fn in os.listdir(model_dir):
            if fn == file_name:
                return True
        return False

    exists = prev_data_exists(MODELS_DIR)

    if not exists:
        # Initialize population with custom initialization
        population = Model().mutate((0, tp, [], (False, tid)))
        models_info = None
        latest_generation = 0
        
    else:
        # Load the latest generation as an array
        models_data_file = os.path.join(MODELS_DIR, file_name)
        models_info = np.load(models_data_file)
        latest_generation = int(np.max(models_info[:, 1]))

    #for generation in range(utils.get_newest_generation_number(networks_dir) + 1, generations):
    for generation in range(latest_generation + 1, tp["generations"]):

        # Extract the networks of the top performers
        top_models_weights = models_info[:tp["top_n"], 2:]

        population = mutate_next_gen(top_models_weights, tp, profile=(profile, tid))

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
        print_stats(utils.filter_methods(p, current_dir).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)

if __name__ == "__main__":
    
    if pltfm == 'WSL':
        curses.wrapper(main)
    else:
        main(None)