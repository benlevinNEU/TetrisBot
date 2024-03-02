import numpy as np
import os, sys, time, cProfile, multiprocessing, platform, re
from place_tetris import TetrisApp, COLS, ROWS

from multiprocessing import Manager, Pool, Value
import concurrent.futures

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

from local_params import GP, TP
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
    "mutation_strength": lambda gen: 10 * np.exp(-0.001 * gen) + 0.1,
    "profile": True,
    "workers": 0, # Use all available cores
    "feature_transform": lambda x: np.array([x**2, x, 1]),
}
'''

MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

class Model():
    def __init__(self, tp=TP, weights=[]):
        if len(weights) == 0:
            weights = np.ones(tp["feature_transform"](0).shape[0]*NUM_EVALS)
        self.weights = weights.reshape(tp["feature_transform"](0).shape[0], NUM_EVALS)

    def play(self, gp, pos, tp=TP):
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

                c = self.cost(option, tp)
                if c < min_cost:
                    min_cost = c
                    best_option = option

            options, game_over, score = self.game.ai_command(best_option)

        self.game.quit_game()
        return score

    def cost(self, state, tp):
        vals = np.array(getEvals(state))
        X = np.array([tp["feature_transform"](x) for x in vals])
        return np.sum(X.T @ self.weights.T)

    def mutate(self, gen):
        
        children = []
        nchildren = int(TP["population_size"] / TP["top_n"])

        for _ in range(nchildren):
            new_weights = self.weights.copy()

            for i in range(len(new_weights)):
                if np.random.rand() < TP["mutation_rate"](gen):
                    new_weights[i] += TP["mutation_strength"](gen) * (np.random.randn()*2 - 1) # Can increase or decrease weights

            children.append(Model(TP, new_weights))

        return children

    def evaluate(self, args):
        id, plays, gp = args

        # Won't always put window in same place bc proccesses will finish in unknown order
        slot = id % MAX_WORKERS

        width = gp["cell_size"] * (gp["cols"] + 6)
        height = gp["cell_size"] * gp["rows"] + 80
        pos = ((width * slot) % 2560, height * int(slot / int(2560 / width)))

        scores = []
        for _ in range(plays):
            scores.append(self.play(gp, pos))

        return (self.weights.flatten(), np.mean(scores))

def mutate_model(args):
    weights, _, gen = args # id is not used
    model = Model(weights=weights)
    return model.mutate(gen)

def evaluate_model(args):
    model, _, _, _ = args
    return model.evaluate(args[1:])

def profiler_wrapper(args):
    """
    Wrapper function to profile a task function.

    Args:
    - task_func: The original task function to profile.
    - task_args: Arguments to pass to the task function.
    - profile_dir: Directory to save the profile data.
    - profile_prefix: Prefix for the profile data filename.
    """

    model, id, task_func, profile_prefix, *task_args = args

    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute the original task function
    # id needs to be passed to the task function bc was previously appended to end of args
    ret = task_func((model, id, *task_args)) # TODO: Check if this is the correct way to pass args
    
    profiler.disable()
    timestamp = int(time.time())
    profile_path = os.path.join(PROF_DIR, f"{profile_prefix}/proc_{timestamp}.prof")
    profiler.dump_stats(profile_path)

    return ret

def pool_task_wrapper(task_func, task_args, profile, prnt_lbl):
    """
    Wrapper function to execute a task function in a pool of worker processes.

    Args:
    - task_func: The original task function to execute.
    - task_args: Arguments to pass to the task
        - population of tasks to execute (either list of models or list of model weights)
        - rest of args
    - profile: Tuple of (bool, int) indicating whether to profile the task and the profile prefix.
    - prnt_lbl: Label for the task to print progress.
    """

    ret_list = []

    # Check if profiling is enabled
    if profile[0]:
        func = profiler_wrapper
        args = (task_args[0], task_func, profile[1], *task_args[1:])
    else:
        func = task_func
        args = task_args

    # Execute the task function
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Grabs the models from the first arg and assigns an id to each model before submitting
        futures = [executor.submit(func, (model, id, *args[1:])) for id, model in enumerate(args[0])]

        # Track progress and collect results
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            progress = (count / len(args[0])) * 100
            # Prints the progress percentage with appropriate task label
            sys.stdout.write(f"{prnt_lbl}: {progress:.2f}%             \r") # Overwrites the line
            sys.stdout.flush()
            ret_list.append(future.result())

    return ret_list

def main(stdscr):

    ## General Setup
    profile = TP["profile"]
    tid = int(time.time())
    profiler_dir = f"{PROF_DIR}{tid}/"

    if profile:
        os.makedirs(profiler_dir)
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
    nft = TP["feature_transform"](0).shape[0] # Number of fe transforms
    file_name = f"models_{GP['rows']}x{GP['cols']}_{nft}.npy"
    print(f"Saving data to: {MODELS_DIR}{file_name}")

    def prev_data_exists(model_dir):
        for fn in os.listdir(model_dir):
            if fn == file_name:
                return True
        return False

    exists = prev_data_exists(MODELS_DIR)

    if not exists:
        # Initialize population with custom initialization
        init_tp = TP.copy()
        init_tp["top_n"] = 1
        population = Model(tp=init_tp).mutate(0)
        models_info = None
        latest_generation = 0

    else:
        # Load the latest generation as an array
        models_data_file = os.path.join(MODELS_DIR, file_name)
        models_info = np.load(models_data_file)
        latest_generation = int(np.max(models_info[:, 1]))

    for generation in range(latest_generation + 1, TP["generations"]):

        if models_info is None:
            # Evaluate all networks in parallel
            results = pool_task_wrapper(evaluate_model, (population, TP["plays"], GP), (profile, tid), "Generation Running")

            # Unpack the results and create a numpy array with score in the first column and weights after it
            models_info = np.array([(score,generation) + tuple(weights) for weights, score in results])

            # Select top performers
            models_info = models_info[models_info[:, 0].argsort()[::-1]]
            
        top_models_weights = models_info[:TP["top_n"], 2:] # 2 Labels before weights (score, generation)

        population = np.array(pool_task_wrapper(mutate_model, (top_models_weights, generation), (profile, tid), "Mutating")).flatten().tolist()

        # Evaluate all networks in parallel
        results = pool_task_wrapper(evaluate_model, (population, TP["plays"], GP), (profile, tid), "Generation Running")

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
        exists = True
        
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
        profiler.dump_stats(f"{profile_dir}main.prof")

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)

if __name__ == "__main__":
    
    if pltfm == 'WSL':
        curses.wrapper(main)
    else:
        main(None)