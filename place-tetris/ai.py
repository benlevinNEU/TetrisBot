from math import pi
import numpy as np
import pandas as pd
from scipy import stats
import os, sys, time, cProfile, multiprocessing, platform, re
from place_tetris import TetrisApp

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
GP = {
    "gui": False,  # Set to True to visualize the game
    "cell_size": 20,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 0.01
}

# Initialize the training parameters
TP = {
    "population_size": 100,
    "top_n": 20,
    "generations": 1000,
    "max_plays": 30,
    "mutation_rate": lambda gen: 0.5 * np.exp(-0.002 * gen) + 0.1,
    "mutation_strength": lambda gen: 0.3 * np.exp(-0.002 * gen) + 0.1,
    "s_mutation_strength": lambda gen: 0.1 * np.exp(-0.002 * gen) + 0.02,
    "momentum": [0.9, 0.1],
    "profile": True,
    "workers": 8,
    "feature_transform": "x,1/(x+0.1),np.ones_like(x)",
    "learning_rate": lambda gen: 0.8 * np.exp(-0.002 * gen) + 0.1,
    "s_learning_rate": lambda gen: 0.1 * np.exp(-0.002 * gen) + 0.1,
    "age_factor": lambda age: 0.1 * np.exp(0.05 * age) + 1,
    "p_random": 0.1
}
'''

FT = eval(f"lambda self, x: np.column_stack([{te.decode(TP["feature_transform"])}])")
MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

class Model():
    def __init__(self, tp=TP, weights=[], sigmas=np.random.uniform(0.01, 0.99, NUM_EVALS), gen=1):
        self.sigma = sigmas
        ft = eval(f"lambda self, x: np.array([{te.decode(tp["feature_transform"])}])")
        self.fts = ft(self, np.ones(NUM_EVALS)).shape[0]

        if len(weights) == 0:
            weights = np.random.uniform(-15, 15, [NUM_EVALS, self.fts])

        self.weights = weights
        self.gen = gen
        self.tp = tp

    def play(self, gp, pos, tp=TP):
        game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos)
        
        if gp["gui"]:
            game.update_board()

        options = game.getFinalStates()
        gameover = False
        score = 0
        tot_cost = 0
        moves = 0
        norm_c_grad = np.zeros([NUM_EVALS, self.fts])
        norm_s_grad = np.zeros(NUM_EVALS)

        eval_labels = getEvalLabels()
        #print(" ".join(eval_labels)) # TODO: Comment out

        while not gameover and len(options) > 0 and moves < 10000: # Ends the game after 10000 moves
            min_cost = np.inf
            best_option = None

            for option in options:

                if option is None:
                    raise ValueError("Option is None")

                c, w_grad, s_grad = self.cost(option, tp)
                if c < min_cost:
                    min_cost = c
                    best_option = option
                    min_w_grad = w_grad
                    min_s_grad = s_grad

            #print(" ".join(f"{val:.5f}" for val in w_grad[:,0]))  # TODO: Comment out
                    
            tot_cost += min_cost
            norm_c_grad += min_w_grad

            if min_s_grad is None:
                norm_s_grad = None
            else:
                norm_s_grad += min_s_grad

            moves += 1
            options, game_over, score = game.ai_command(best_option)

        # Return the absolute value of the average cost per move and the average gradient
        w_cost_metrics = norm_c_grad/moves/score

        if norm_s_grad is None:
            s_cost_metrics = None
        else:
            s_cost_metrics = norm_s_grad/moves/score

        if moves == 10000:
            success_log = open(F"{CURRENT_DIR}success.log", "a")
            success_log.write("Game ended after 10000 moves\n")
            success_log.write(f"{self.weights}")
            success_log.close()

        game.quit_game()
        return score, w_cost_metrics, s_cost_metrics
    
    def gauss(self, x, mu=0.5):
        #print(f"sigma: {self.sigma}")
        #selfprint(f"x: {x}")
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * self.sigma**2))

    def sigma_grad(self, x, mu=0.5):
        prefactor = -1 / (self.sigma**2) + ((x - mu)**2) / (self.sigma**4)
        gaussian = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((x - mu) / self.sigma)**2)
        return prefactor * gaussian

    def cost(self, state, tp=TP):
        vals = getEvals(state)
        X = FT(self, vals)
        costs = X * self.weights

        sigma_grad = None
        if "self.gauss" in tp["feature_transform"]:
            gWeights = self.weights[:, tp["feature_transform"].index("self.gauss")]
            sigma_grad = self.sigma_grad(vals) * gWeights

        return np.sum(costs), X, sigma_grad

    def mutate(self, gen, w_grad=None, s_cm=None):

        # TODO: Introduce momentum factor

        if w_grad is None:
            w_grad = np.ones([NUM_EVALS, self.fts])

        if s_cm is None:
            s_cm = np.ones(NUM_EVALS)

        # Regularize gradient step scale largest step to the learning rate
        w_l2 = np.linalg.norm(self.weights)
        w_step = TP["learning_rate"](gen) * w_l2 * w_grad/np.linalg.norm(w_grad)

        s_l2 = np.linalg.norm(self.sigma)
        s_step = TP["s_learning_rate"](gen) * s_l2 * s_cm/np.linalg.norm(s_cm)
        
        nchildren = int(TP["population_size"] * (1 - TP["p_random"]) / TP["top_n"])
        children = []
        age_factor = min(100, TP['age_factor'](gen - self.gen))

        # Regularize mutation
        w_strength = TP["mutation_strength"](gen) * w_l2
        s_strength = TP["s_mutation_strength"](gen) * s_l2
        for _ in range(nchildren):
            new_weights = self.weights.copy()

            # Check if bias term is included in feature transform
            index = None
            if "np.ones_like(x)" in self.tp["feature_transform"]:
                bias_start = self.tp["feature_transform"].index("np.ones_like(x)")
                index = np.sum([1 for char in self.tp["feature_transform"][:bias_start] if char == ','])

                # Excludes bias term from stepping gradient
                new_weights[:, np.arange(new_weights.shape[1]) != index] += w_step[:, np.arange(new_weights.shape[1]) != index]

            else:
                new_weights += w_step

            new_sigmas = self.sigma.copy()
            new_sigmas -= s_step

            # Random mutation introduction
            for e in range(NUM_EVALS):
                for f in range(self.fts):
                    if np.random.rand() < TP["mutation_rate"](gen):
                        # Strong mutation if weight is 0
                        if new_weights[e,f] == 0:
                            strength = w_strength * 10
                        else:
                            strength = w_strength

                        # Strengthen mutation for bias
                        if (index is not None) and (f == index):
                            strength *= 5

                        # Strengthen mutation for very old parents (max age factor is 100)
                        age_factor = min(100, TP['age_factor'](gen - self.gen))
                        mutation = np.random.normal(0, strength, 1)[0] # Can increase or decrease weights
                        
                        new_weights[e,f] += mutation * age_factor

            for i in range(len(new_sigmas)):
                if np.random.rand() < TP["mutation_rate"](gen):
                    new_sigmas[i] += age_factor * np.random.normal(0, s_strength, 1)[0]

            new_sigmas = np.clip(new_sigmas, 0.01, 0.99)

            model = Model(weights=new_weights, sigmas=new_sigmas)
            model.tp = None # Remove reference to TP for pickling
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
        w_cost_metrics_lst = np.zeros((plays,shape[0],shape[1])) 
        s_cost_metrics_lst = np.zeros((plays, shape[0]))
        for i in range(plays):
            score, w_cost_metrics, s_cost_metrics = self.play(gp, pos, TP)
            scores[i] = score
            w_cost_metrics_lst[i] = w_cost_metrics
            s_cost_metrics_lst[i] = s_cost_metrics

            if not playMore(scores):
                break

        # Trim to only played games before calculating expected score
        score = expectedScore(scores[:i+1])
        # TODO: Should I trim gradients as well or just the score?
        s_cost_metrics = np.mean(s_cost_metrics_lst, axis=0)

        df = pd.DataFrame({
            'score': [score],
            'model': [self],
            'weights': [self.weights.flatten()],
            'sigmas': [self.sigma],
            'w_cost_metrics': [w_cost_metrics.flatten()],
            's_cost_metrics': [s_cost_metrics],
        })

        return df

# Method to play more games if the standard deviation is not stable
# TODO: Might need to develop algo to tune theshold value
def playMore(scores, threshold=0.003, max_count=TP["max_plays"]):

    if len(scores) < 10:
        return max_count  # Not enough data to make a decision

    new_std = np.std(scores)
    prev_std = np.std(scores[:-1])
    if abs(new_std - prev_std) / prev_std < threshold:
        return False  # The number of games where the estimate stabilized

    return True  # Return the max games if the threshold is never met

# Method to calculate expected score
# Scores from models have high stdev and mean has heavy right skew
def expectedScore(scores):

    x = np.linspace(np.min(scores), np.max(scores), 100)
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)
    #p_lognorm = stats.lognorm.pdf(x, shape_lognorm, loc_lognorm, scale_lognorm)
    med_lognorm = stats.lognorm.ppf(0.5, shape_lognorm, loc_lognorm, scale_lognorm)
    return med_lognorm

    '''
    std = np.std(scores)
    samples = len(scores)

    # Base prop
    base_prop = 0.2

    # TODO: WIll need to tune this bc prop is never being adjusted bc std is always very large

    # Adjust the proportion based on the standard deviation and sample size
    # This is a simple heuristic and can be adjusted based on empirical testing
    prop = base_prop * std / np.sqrt(samples)

    # Ensure the proportion is within a sensible range, e.g., 0.01 to 0.25
    prop = max(0.01, min(prop, 0.25))

    return np.mean(stats.mstats.winsorize(scores, limits=(prop, prop)))
    '''

# Method wrapper
def mutate_model(args):
    model_df, _, gen = args # id is not used
    weights = model_df['weights'].reshape([NUM_EVALS, int(len(model_df['weights'])/NUM_EVALS)])
    model = Model(weights=weights, sigmas=model_df['sigmas'], gen=model_df['gen'])
    w_cm = model_df['w_cost_metrics'].reshape([NUM_EVALS, model.fts])
    return model.mutate(gen, w_cm, model_df['s_cost_metrics'])

# Method wrapperop  _ste[]
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
            progress = (count / len(args[0])) * 100

            # Prints the progress percentage with appropriate task label
            e_time = time.time() - start_time
            sys.stdout.write(f"{prnt_lbl}: {progress:.2f}%  Elapsed Time / Estimate (s): {e_time:.0f}/{e_time/progress *100:.0f}             \r") # Overwrites the line
            sys.stdout.flush()

    else:
        # Execute the task function
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Grabs the models from the first arg and assigns an id to each model before submitting
            start_time = time.time()  # Assign a valid value to start_time
            futures = [executor.submit(
                func, (model, id, *args[1:])) for id, model in args[0].iterrows()]

            # Track progress and collect results
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                progress = (count / len(args[0])) * 100
                # Prints the progress percentage with appropriate task label
                e_time = time.time() - start_time
                sys.stdout.write(f"{prnt_lbl}: {progress:.2f}%  Elapsed Time / Estimate (s): {e_time:.0f}/{e_time/progress *100:.0f}             \r") # Overwrites the line
                sys.stdout.flush()
                ret_list.append(future.result())

    # Write elapsed time such that it isn't overritten by the generation number and score once leaving the pool task wrapper
    # I did this so I didn't have to handle packaging it in the result
    sys.stdout.write(f"                                   Elapsed Time (s): {e_time:.0f}             \r") # Overwrites the line
    sys.stdout.flush()

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
            #population = Model(tp=init_tp).mutate(0)
            
            # Create randomly initialized population
            population = pd.DataFrame([], columns=['model'])
            for _ in range(TP["population_size"]):
                model = Model(tp=init_tp)
                model.tp = None # Remove reference to TP for pickling
                population = pd.concat([population, pd.DataFrame([model], columns=['model'])], ignore_index=True)

            saved_models_info = pd.DataFrame([])

        else:
            # Load the latest generation as an array
            models_data_file = os.path.join(MODELS_DIR, file_name)
            #models_info = np.load(models_data_file)
            saved_models_info = pd.read_parquet(models_data_file)
            generation = int(np.max(saved_models_info['gen'])) + 1
            
            top_models = saved_models_info.head(TP["top_n"]) # 2 Labels before weights (score, generation)
            population = pd.concat(pool_task_wrapper(
                mutate_model, (top_models, generation), (profile if MAX_WORKERS > 1 else False, tid), "Mutating"), ignore_index=True)

            rand_restart = int(TP["population_size"] * TP["p_random"])
            for _ in range(rand_restart):
                model = Model(gen=generation)
                model.tp = None # Remove reference to TP for pickling
                population = pd.concat([population, pd.DataFrame([model], columns=['model'])], ignore_index=True)

        # Evaluate all networks in parallel
        results = pool_task_wrapper(
            evaluate_model, (population, TP["max_plays"], GP), (profile if MAX_WORKERS > 1 else False, tid), "Generation Running")
        raw_df = pd.concat(results, ignore_index=True)
        raw_df["gen"] = generation
        raw_df = raw_df.sort_values(by="score", ascending=False)

        # Select top score
        top_score = raw_df.iloc[0]["score"]
        print(f"Generation {generation}: Top Score = {top_score:.1f}\r")

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