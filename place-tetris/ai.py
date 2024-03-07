import numpy as np
import pandas as pd
import os, sys, time, cProfile, multiprocessing, platform, re
from place_tetris import TetrisApp, COLS, ROWS

from multiprocessing import Manager, Pool, Value
import concurrent.futures

import utils
from get_latest_profiler_data import print_stats
from evals import *
import transform_encode as te

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
    "feature_transform": "x**2,x,1",
}
'''

FT = eval(f"lambda x: np.array([{te.decode(TP["feature_transform"])}])")
MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

class Model():
    def __init__(self, tp=TP, weights=[]):
        ft = eval(f"lambda x: np.array([{te.decode(tp["feature_transform"])}])")
        if len(weights) == 0:
            weights = np.ones(ft(0).shape[0]*NUM_EVALS)
        self.weights = weights.reshape(ft(0).shape[0], NUM_EVALS)

    def play(self, gp, pos, tp=TP):
        self.game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos)
        
        if gp["gui"]:
            self.game.update_board()

        options = self.game.getFinalStates()
        gameover = False
        score = 0
        tot_cost = 0
        moves = 0
        norm_c_grad = np.zeros(NUM_EVALS*(FT(0).shape[0]-1))

        while not gameover and len(options) > 0 and moves < 10000: # Ends the game after 10000 moves
            min_cost = np.inf
            best_option = None

            for option in options:

                if option is None:
                    raise ValueError("Option is None")

                c, c_grad = self.cost(option, tp)
                if c < min_cost:
                    min_cost = c
                    best_option = option
                    min_c_grad = c_grad

            tot_cost += min_cost
            norm_c_grad += min_c_grad
            moves += 1
            options, game_over, score = self.game.ai_command(best_option)

        cost_metrics = np.array([tot_cost/moves/score, *norm_c_grad/moves/score])

        if moves == 10000:
            success_log = open(F"{CURRENT_DIR}success.log", "a")
            success_log.write("Game ended after 10000 moves\n")
            success_log.write(f"{self.weights}")
            success_log.close()

        self.game.quit_game()
        return score, cost_metrics

    def cost(self, state, tp):
        vals = np.array(getEvals(state))
        X = np.array([FT(x) for x in vals])
        costs = X * self.weights.T
        return np.sum(costs), X[:,:-1].flatten() # Exclude bias term because not relevant for gradient descent

    def mutate(self, gen, cm=None):

        FT_s = FT(0).shape[0]
        if cm is None:
            cm = np.ones((FT_s-1) * NUM_EVALS + 1)

        grad_mag = cm[0]
        grad = cm[1:].reshape(FT_s-1, NUM_EVALS)
        step = TP["learning_rate"](gen) * grad/grad_mag
        
        nchildren = int(TP["population_size"] / TP["top_n"])
        children = []
        for _ in range(nchildren):
            new_weights = self.weights.copy()

            new_weights[0:FT_s-1, 0:NUM_EVALS] += step # Can increase or decrease weights
            flat_weights = new_weights.flatten()

            for i in range(len(flat_weights)):
                if np.random.rand() < TP["mutation_rate"](gen):
                    if flat_weights[i] == 0:
                        strength = TP["mutation_strength"](gen) * 2
                    elif (i+1) % FT_s != 0:
                        strength = TP["mutation_strength"](gen) * 0.1
                    else:
                        strength = TP["mutation_strength"](gen)
                    flat_weights[i] += strength * (np.random.randn()*2 - 1) # Can increase or decrease weights

            model = Model(weights=flat_weights)
            children.append(model)

        return pd.DataFrame(children, columns=['model'])

    def evaluate(self, args):
        id, plays, gp = args

        # Won't always put window in same place bc proccesses will finish in unknown order
        slot = id % MAX_WORKERS

        width = gp["cell_size"] * (gp["cols"] + 6)
        height = gp["cell_size"] * gp["rows"] + 80
        pos = ((width * slot) % 2560, height * int(slot / int(2560 / width)))

        scores = np.zeros(plays)
        shape = self.weights.shape
        # 1 for score, rest for cost metrics not including bias term
        cost_metrics_lst = np.zeros((plays, 1 + shape[1]*(shape[0]-1))) 
        for i in range(plays):
            score, cost_metrics = self.play(gp, pos, TP)
            scores[i] = score
            cost_metrics_lst[i] = cost_metrics

        score = np.mean(scores)
        cost_metrics = np.mean(cost_metrics_lst, axis=0)

        df = pd.DataFrame({
            'score': [score],
            'model': [self],
            'weights': [self.weights.flatten()],
            'cost_metrics': [cost_metrics]
        })

        return df

# Method wrapper
def mutate_model(args):
    model_df, _, gen = args # id is not used
    model = Model(weights=model_df['weights'])
    return model.mutate(gen, model_df['cost_metrics'])

# Method wrapperop  
def evaluate_model(args):
    model_df, _, _, _ = args
    return model_df['model'].evaluate(args[1:])

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
    ret = task_func((model, id, *task_args))
    
    profiler.disable()
    timestamp = int(time.time())
    profile_path = os.path.join(PROF_DIR, f"{profile_prefix}/proc_{timestamp}.prof")
    profiler.dump_stats(profile_path)

    return ret

def pool_task_wrapper(task_func, task_args, profile, prnt_lbl):
    """turn
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

    # For easier debugging
    # Raper method hides source of thrownxceptions
    if MAX_WORKERS == 1: 
        # Execute the task function sequentiallyodel
        for id, model in args[0].iterrows():
            ret_list.append(func((model, id, *args[1:])))
    else:
        # Execute the task function
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Grabs the models from the first arg and assigns an id to each model before submitting
            futures = [executor.submit(
                func, (model, id, *args[1:])) for id, model in args[0].iterrows()]

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

    exit_event = multiprocessing.Event()

    def on_press(key):
        # Check if the pressed key is ESC
        if key == Key.esc:
            exit_event.set()
            return False

    # Set up the listener
    if pltfm == 'Mac':
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    elif pltfm == 'WSL':
        # Initialize curses environment
        curses.cbreak()  # Disable line buffering
        stdscr.keypad(True)  # Enable special keys to be recorded
        curses.noecho()  # Prevent input from being echoed to the screen
        stdscr.nodelay(True)  # Make getch() non-blocking


    print("Starting Evolutionary Training")


    ## Load or Initialize Population
    ft = TP["feature_transform"] # Number of feature transforms
    #file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.npy"
    file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
    print(f"\rSaving data to: \n\r{MODELS_DIR}{file_name}\n\r")

    def prev_data_exists(model_dir):
        for fn in os.listdir(model_dir):
            if fn == file_name:
                return True
        return False

    exists = prev_data_exists(MODELS_DIR)
    generation = 0

    for generation in range(0, TP["generations"]):

        if not exists:
            init_tp = TP.copy()
            init_tp["top_n"] = 1
            population = Model(tp=init_tp).mutate(0)
            saved_models_info = pd.DataFrame([])

        else:
            # Load the latest generation as an array
            models_data_file = os.path.join(MODELS_DIR, file_name)
            #models_info = np.load(models_data_file)
            saved_models_info = pd.read_parquet(models_data_file)
            generation = int(np.max(saved_models_info['gen'])) + 1
            
            top_models = saved_models_info.head(TP["top_n"]) # 2 Labels before weights (score, generation)
            population = pd.concat(pool_task_wrapper(
                mutate_model, (top_models, generation), (profile, tid), "Mutating"), ignore_index=True)

        # Evaluate all networks in parallel
        results = pool_task_wrapper(
            evaluate_model, (population, TP["plays"], GP), (profile, tid), "Generation Running")
        raw_df = pd.concat(results, ignore_index=True)
        raw_df["gen"] = generation
        raw_df = raw_df.sort_values(by="score", ascending=False)

        # Select top score
        top_score = raw_df.iloc[0]["score"]
        print(f"Generation {generation}: Top Score = {top_score}\r")

        # Unpack the results and create a numpy array with score in the first column and weights after it
        models_info = pd.concat([saved_models_info, raw_df])

        # Sort by preformance
        models_info = models_info.sort_values(by="score", ascending=False)
        del models_info["model"]

        # Save the numpy array to the file
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        save_file = os.path.join(MODELS_DIR, file_name)

        models_info.to_parquet(save_file)
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
        if exit_event.is_set():
            break

    print("Finished Evolutionary Training")


    ## Clean Up
    if profile:
        profiler.disable()
        profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)

if __name__ == "__main__":
    
    if pltfm == 'WSL':
        curses.wrapper(main)
    else:
        main(None)