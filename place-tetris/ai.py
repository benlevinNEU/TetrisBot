from math import pi
import numpy as np
import pandas as pd
from scipy import stats
import os, sys, time, cProfile, multiprocessing, platform, re
from place_tetris import TetrisApp, trimBoard

from multiprocessing import Manager, Pool, Value
import concurrent.futures

import utils, importlib
from get_latest_profiler_data import print_stats
from evals import Evals, NUM_EVALS, getEvalLabels
import transform_encode as te

import tqdm

pltfm = None
if platform.system() == 'Linux' and 'microsoft-standard-WSL2' in platform.release():
    pltfm = 'WSL'
    import curses
    #import keyboard
else:
    pltfm = 'Mac'
    from pynput import keyboard # type: ignore
    from pynput.keyboard import Key # type: ignore

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import local_params
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
    "top_n": 10,
    "generations": 1000,
    "max_plays": 20,
    "mutation_rate": lambda gen: 0.1 * np.exp(-0.003 * gen) + 0.05,
    "mutation_strength": lambda gen: 0.001 * np.exp(-0.005 * gen) + 0.0001,
    "s_mutation_strength": lambda gen: 0.001 * np.exp(-0.005 * gen) + 0.0001,
    "momentum": [0.9, 0.1],
    "profile": False,
    "workers": 0,
    "feature_transform": "x",
    "learning_rate": lambda gen: 0.01 * np.exp(-0.005 * gen) + 0.001,
    "s_learning_rate": lambda gen: 0.01 * np.exp(-0.005 * gen) + 0.001,
    "age_factor": lambda age: 0.05 * np.exp(0.003 * age) + 1,
    "p_random": 0.1,
    "prune_ratio": 0.3,
    "cutoff": 300,
}
'''

FT = eval(f"lambda self, x: np.column_stack([{te.decode(TP["feature_transform"])}])")
MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

class Model():
    def __init__(self, tp=TP, weights=[], sigmas=np.random.uniform(0.01, 0.99, NUM_EVALS), gen=1, parent_gen=0, parentID=None, id=None):
        self.sigma = sigmas
        ft = eval(f"lambda self, x: np.array([{te.decode(tp["feature_transform"])}])")
        self.fts = ft(self, np.ones(NUM_EVALS)).shape[0]

        if len(weights) == 0 :
            weights = np.random.uniform(-15, 15, [NUM_EVALS, self.fts])

        weights = np.array(weights, copy=True)
        zero_indices = np.where(~weights.all(axis=0))[0]
        weights[:, zero_indices] = 0.01 * weights[:, 0].reshape(-1, 1)

        self.weights = weights
        self.gen = gen
        self.tp = tp
        self.parent_gen = parent_gen
        self.parentID = parentID
        if id is None:
            self.id = str(hash((gen, str(weights))))
        else:
            self.id = id
        
        self.snapshots = []

    def play(self, gp, pos, tp=TP, ft=FT, use_snap=True):

        # Loop to initialize game until a valid starting state is found if starting from snap
        while True:
            if np.random.rand() < tp['use_snap_prob'] and len(self.snapshots) > 0 and use_snap:
                choice = np.random.choice(np.arange(len(self.snapshots)))
                snapshot = self.snapshots[choice]

                self.snapshots.pop(choice)
                snapscore = snapshot[1]
                game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos, snap=snapshot)

                snap_log = open(f"{CURRENT_DIR}/snap.log", "a")
                snap_log.write(f"Model ID: {self.id}\n")
                snap_log.write(f"Snapshot Used\n")
                snap_log.write(f"Score: {snapshot[1]}\n")
                snap_log.write(f"{trimBoard(snapshot[0])}\n")
                snap_log.close()

            else:
                snapscore = 0
                snapshot = None
                game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos)
            
            if gp["gui"]:
                game.update_board()

            options = game.getFinalStates()

            if len(options) > 0:
                break

        gameover = False
        score = 0
        tot_cost = 0
        moves = 0
        norm_c_grad = np.zeros([NUM_EVALS, self.fts])
        norm_s_grad = np.zeros(NUM_EVALS)

        eval_labels = getEvalLabels()
        #print(" ".join(eval_labels)) # TODO: Comment out

        while not gameover and len(options) > 0 and (tp['demo'] or moves < tp['cutoff']): # Ends the game after CUTOFF moves unless demo
            min_cost = np.inf
            best_option = None

            for option in options:

                if option is None:
                    raise ValueError("Option is None")

                c, w_grad, s_grad, mxh = self.cost(option, tp, ft)
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
            options, game_over, score, snapshot = game.ai_command(best_option, cp=(self, tp, ft))

            if score == 0:
                raise ValueError("Score is 0")

            # Save snapshot of the game state
            if np.random.rand() < tp['snap_prob'](mxh):
                self.snapshots.append(snapshot)

                snap_log = open(f"{CURRENT_DIR}/snap.log", "a")
                snap_log.write(f"Snapshot Taken\n")
                snap_log.write(f"Model ID: {self.id}\n")
                snap_log.write(f"Score: {snapshot[1]}\n")
                snap_log.write(f"{trimBoard(snapshot[0])}\n")
                snap_log.close()

        # In the event that no more points are scored
        if score - snapscore == 0:
            return score, np.zeros([NUM_EVALS, self.fts]), np.zeros(NUM_EVALS)

        # Return the absolute value of the average cost per move and the average gradient
        w_cost_metrics = norm_c_grad/moves/(score - snapscore) 

        if norm_s_grad is None:
            s_cost_metrics = None
        else:
            s_cost_metrics = norm_s_grad/moves/(score - snapscore)

        if moves == tp['cutoff']:
            success_log = open(f"{CURRENT_DIR}/success.log", "a")
            success_log.write(f"Game ended after {tp['cutoff']} moves\n")
            success_log.write(f"{self.weights}\n")
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

    def cost(self, state, tp=TP, ft=FT):
        vals = self.evals.getEvals(state)
        X = ft(self, vals)
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        costs = X * self.weights

        sigma_grad = None
        if "self.gauss" in tp["feature_transform"]:
            ft_start = tp["feature_transform"].index("self.gauss")
            index = np.sum([1 for char in tp["feature_transform"][:ft_start] if char == ','])
            gWeights = self.weights[:, index]
            sigma_grad = self.sigma_grad(vals) * gWeights

        mxh = vals[1] # Save max height for snapshot

        return np.sum(costs), X, sigma_grad, mxh

    def evaluate(self, args):
        id, plays, gp = args

        self.evals = Evals(gp)

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

            if not playMore(scores[:i+1]):
                break

        # Trim to only played games before calculating expected score
        scores = scores[:i+1]
        score, ln_vals = expectedScore(scores, True)

        # Improve weighting of cost gradient of poorly preforming plays by preforming a log-norm transform on the metrics 
        # based on how they are distributed among the play throughs and summing the results of the transform
        w = stats.lognorm.pdf(scores, *ln_vals)

        w_cost_metrics = np.sum(w_cost_metrics_lst[:i+1] * w[:, np.newaxis, np.newaxis], axis=0)
        s_cost_metrics = np.sum(s_cost_metrics_lst[:i+1] * w[:, np.newaxis], axis=0)
        
        std = np.std(scores)

        shape, _, scale = ln_vals

        def rank(shape, scale):
            exp = np.log(scale)
            std = shape
            expected_value = np.exp(exp - std/2)
    
            return expected_value

        def aic(scores):
            shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)
            log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
            return 2*3 - 2*log_likelihood_lognorm

        df = pd.DataFrame({
            'gen': [self.gen],
            'rank': [rank(shape, scale)],
            'exp_score': [score],
            'std': [std],
            'aic': [aic(scores)],
            'shape': [shape],
            'scale': [scale],
            'model': [self],
            'id': [self.id],
            'parentID': [self.parentID],
            'weights': [self.weights.flatten()],
            'sigmas': [self.sigma],
            'w_cost_metrics': [w_cost_metrics.flatten()],
            's_cost_metrics': [s_cost_metrics],
        })

        return df

    def mutate(self, gen, w_grad=None, s_cm=None, max_children=-1):

        # TODO: Introduce momentum factor

        if w_grad is None:
            w_grad = np.ones([NUM_EVALS, self.fts])

        if s_cm is None:
            s_cm = np.ones(NUM_EVALS)

        age_factor = min(100, TP['age_factor'](gen - self.gen))

        # Regularize gradient step scale largest step to the learning rate
        no_step = np.where(np.all(w_grad == 1, axis=0)) # Don't step on columns of grad that are all init as 1 (used when creating new FT)
        w_l2 = np.linalg.norm(self.weights, axis=0)
        w_step = TP["learning_rate"](gen) * w_l2 * w_grad/np.linalg.norm(w_grad, axis=0)
        w_step[:, no_step] = 0

        s_l2 = np.linalg.norm(self.sigma)
        s_step = TP["s_learning_rate"](gen) * s_l2 * s_cm/np.linalg.norm(s_cm)
        
        if max_children == -1: # For lineage limiting
            max_children = TP["population_size"]*TP["lin_prop_max"]
        nchildren = min(int(TP["population_size"] * (1 - TP["p_random"]) / TP["top_n"]), int(max_children))
        children = []

        # Regularize mutation
        w_strength = TP["mutation_strength"](gen) * w_l2
        s_strength = TP["s_mutation_strength"](gen) * s_l2
        for _ in range(nchildren):
            new_weights = self.weights.copy()

            # Age factor for learning rate (randomly increases or decreases learning rate based on age factor)
            lr_age_factor = age_factor if np.random.rand() < 0.5 else 1 / age_factor
            w_step_af = w_step * lr_age_factor

            # Check if bias term is included in feature transform
            index = None
            if "np.ones_like(x)" in self.tp["feature_transform"]:
                bias_start = self.tp["feature_transform"].index("np.ones_like(x)")
                index = np.sum([1 for char in self.tp["feature_transform"][:bias_start] if char == ','])

                # Excludes bias term from stepping gradient
                new_weights[:, np.arange(new_weights.shape[1]) != index] -= w_step_af[:, np.arange(new_weights.shape[1]) != index]

            else:
                new_weights -= w_step_af

            new_sigmas = self.sigma.copy()
            new_sigmas -= s_step

            # Random mutation introduction
            for e in range(NUM_EVALS):
                for f in range(self.fts):
                    if np.random.rand() < TP["mutation_rate"](gen):
                        # Strong mutation if step is 0
                        if w_step[e,f] == 0:
                            strength = w_strength[f] * 500 # Strengthern mutation strength on newly minted FT
                        else:
                            strength = w_strength[f]

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

            model = Model(weights=new_weights, sigmas=new_sigmas, parent_gen=gen, parentID=self.id, tp=self.tp)
            model.tp = None # Remove reference to TP for pickling
            children.append(model)

        return pd.DataFrame(children, columns=['model'])

# Method to play more games if the standard deviation is not stable
# TODO: Might need to develop algo to tune theshold value
def playMore(scores, threshold=0.04, min_count=8, max_count=TP["max_plays"]):

    if len(scores) < min_count:
        return max_count  # Not enough data to make a decision

    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)
    log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    new_aic = 2*3 - 2*log_likelihood_lognorm

    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores[:-1], floc=0)
    log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores[:-1], shape_lognorm, loc_lognorm, scale_lognorm))
    prev_aic = 2*3 - 2*log_likelihood_lognorm

    if abs(new_aic - prev_aic) / prev_aic < threshold:
        return False  # The number of games where the estimate stabilized

    return True  # Return the max games if the threshold is never met

# Method to calculate expected score
# Scores from models have high stdev and mean has heavy right skew
def expectedScore(scores, getVals=False):
    # Fit the log-normal distribution to the data
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)

    # Calculate the expected value (mean) of the log-normal distribution
    # For log-normal, mean = exp(mu + sigma^2 / 2)
    mu = np.log(scale_lognorm)  # scale_lognorm = exp(mu)
    sigma = shape_lognorm
    expected_value = np.exp(mu + (sigma**2) / 2)

    if getVals:
        return expected_value, (shape_lognorm, loc_lognorm, scale_lognorm)

    return expected_value

# Method wrapper
def mutate_model(args):
    model_df, _, gen = args # id is not used
    weights = model_df['weights'].reshape([NUM_EVALS, int(len(model_df['weights'])/NUM_EVALS)])
    model = Model(weights=weights, sigmas=model_df['sigmas'], gen=model_df['gen'], parentID=model_df['parentID'], id=model_df['id'])
    w_cm = model_df['w_cost_metrics'].reshape([NUM_EVALS, model.fts])
    return model.mutate(gen, w_cm, model_df['s_cost_metrics'], model_df['max_children'])

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

    start_time = time.time()  # Assign a valid value to start_time

    # For easier debugging
    # Wrapper method hides source of thrownxceptions
    if MAX_WORKERS == 1: 
        # Execute the task function sequentiallyodel
        for id, model in args[0].iterrows():
            ret_list.append(func((model, id, *args[1:])))
            progress = ((id+1) / len(args[0])) * 100

            # Prints the progress percentage with appropriate task label
            e_time = time.time() - start_time
            sys.stdout.write(f"{prnt_lbl}: {progress:.2f}%  Elapsed Time / Estimate (s): {e_time:.0f}/{e_time/progress *100:.0f}             \r") # Overwrites the line
            sys.stdout.flush()

    else:
        # Execute the task function
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Grabs the models from the first arg and assigns an id to each model before submitting
            futures = [executor.submit(
                func, (model, id, *args[1:])) for id, model in args[0].iterrows()]

            sys.stdout.write(f"\r{prnt_lbl}: {0:.2f}% of {len(args[0])} - Elapsed / Est (s): {0:.0f} / Unknown             ") # Overwrites the line
            sys.stdout.flush()

            # print(prnt_lbl)
            # Track progress and collect results
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            #for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
                progress = (count / len(args[0])) * 100
                # Prints the progress percentage with appropriate task label
                e_time = time.time() - start_time
                sys.stdout.write(f"\r{prnt_lbl}: {progress:.2f}% of {len(args[0])} - Elapsed / Est (s): {e_time:.0f}/{e_time/progress *100:.0f}             ") # Overwrites the line
                sys.stdout.flush()
                ret_list.append(future.result())

    # Write elapsed time such that it isn't overritten by the generation number and score once leaving the pool task wrapper
    # I did this so I didn't have to handle packaging it in the result
    sys.stdout.write(f"\r                                             Elapsed (s): {e_time:.0f}") # Overwrites the line
    sys.stdout.flush()

    return ret_list

def getParents(models_info, pop_size, n_parents, lin_prop_max, p_random):
    # Sort models by descending rank (assuming rank is higher for better models)
    models = models_info.sort_values(by="rank", ascending=False)

    child_set = pop_size * (1 - p_random)
    
    # Initialize lineage tracking and parent selection
    lineage_dict = {}
    lineage_counts = {}
    parents = pd.DataFrame(columns=models_info.columns)
    parents['max_children'] = []

    # Assign each model to its own lineage initially
    for index, model in models.iterrows():
        lineage_dict[model['id']] = set([model['id']])
        lineage_counts[model['id']] = 0

    # Merge lineages based on parent-child relationships
    for index, model in models.iterrows():
        child_id = model['id']
        parent_id = model['parentID']
        if parent_id in models['id'].values:
            lineage_dict[child_id].update(lineage_dict[parent_id])
            for key, lineage in lineage_dict.items():
                if parent_id in lineage:
                    lineage.update(lineage_dict[child_id])

    # Select parents ensuring lineage proportions are maintained
    total_selected = 0
    for index, model in models.iterrows():
        if n_parents >= n_parents and total_selected >= child_set:
            break
        
        current_id = model['id']
        # Identify the lineage of the current model
        current_lineage = lineage_dict[current_id]

        # Calculate the total offspring allowed for the lineage
        lineage_limit = int(np.ceil(lin_prop_max * child_set))

        # Sum the current offspring count for all members of this lineage
        current_lineage_offspring = sum([lineage_counts[id] for id in current_lineage])

        # Determine available slots for this lineage
        available_slots = min(lineage_limit - current_lineage_offspring, int(np.ceil(child_set / n_parents)))

        if available_slots == 0:
            # Reduce the 'max_children' value for all models in the current lineage to accomadate the new model
            '''for _, parent in parents.iterrows():
                if parent['id'] in current_lineage:
                    parent['max_children'] = int(parent['max_children'] * len(parents) / (len(parents) + 1))'''

            lineage_parent_indices = parents[parents['id'].isin(current_lineage)].index
            parents.loc[lineage_parent_indices, 'max_children'] = np.ceil((parents.loc[lineage_parent_indices, 'max_children'] * len(parents) / (len(parents) + 1))).astype(int)

            # Sum the current offspring count for all members of this lineage
            current_lineage_offspring = sum([lineage_counts[id] for id in current_lineage])
            if current_lineage_offspring >= lineage_limit:
                child_set += 1

            # Enable an extra child to be selected for this lineage if lineage is full
            lineage_limit += 1

            # Determine available slots for this lineage
            available_slots = lineage_limit - current_lineage_offspring

            # Flexible number of parents to accomadate all best preforming parents by reducing number of kids per parent
            n_parents += 1
        
        if available_slots > 0:
            selected_slots = int(min(available_slots, child_set - total_selected))
            # Append to parents DataFrame
            new_model = model.copy()
            new_model['max_children'] = selected_slots
            new_model_df = pd.DataFrame([new_model])
            if parents.empty:
                parents = new_model_df
            else:
                parents = pd.concat([parents, new_model_df])
            
            # Update lineage counts
            for id in current_lineage:
                lineage_counts[id] += selected_slots
            total_selected += selected_slots

    return parents

def main(stdscr):

    global TP, GP, FT, MAX_WORKERS

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
    press_count = multiprocessing.Value('i', 0)
    press_time = multiprocessing.Value('d', 0.0)

    def on_press(key):
        # Check if the pressed key is ESC
        if key == Key.esc:
            with press_count.get_lock():
                current_time = time.time()
                if current_time - press_time.value > 3:
                    press_count.value = 1
                    press_time.value = current_time
                else:
                    press_count.value += 1
                if press_count.value >= 5:
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

        importlib.reload(local_params)
        FT = eval(f"lambda self, x: np.column_stack([{te.decode(TP["feature_transform"])}])")
        MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()

        # Clear the snap log
        snap_log = open(f"{CURRENT_DIR}/snap.log", "w")
        snap_log.write(f"")
        snap_log.close()

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
            
            top_models = getParents(saved_models_info, TP['population_size'], TP['top_n'], TP['lin_prop_max'], TP['p_random'])

            population = pd.concat(pool_task_wrapper(
                mutate_model, (top_models, generation), (profile if MAX_WORKERS > 1 else False, tid), "Mutating"), ignore_index=True)

            rand_restart = int(TP["population_size"] * TP["p_random"])
            for _ in range(rand_restart):
                model = Model(gen=generation)
                model.tp = None # Remove reference to TP for pickling
                population = pd.concat([population, pd.DataFrame([model], columns=['model'])], ignore_index=True)

        # Evaluate all networks in parallel
        results = pool_task_wrapper(
            evaluate_model, (population, TP["max_plays"], GP), (profile if MAX_WORKERS > 1 else False, tid), "Gen Running")
        raw_df = pd.concat(results, ignore_index=True)
        raw_df["gen"] = generation
        raw_df = raw_df.sort_values(by="rank", ascending=False)

        # Select top score
        top_score = raw_df.iloc[0]["exp_score"]
        print(f"\rGeneration {generation}: Top Expected Score = {top_score:.1f}")

        # Unpack the results and create a numpy array with score in the first column and weights after it
        models_info = pd.concat([saved_models_info, raw_df])

        # Sort by preformance
        models_info = models_info.sort_values(by="rank", ascending=False)
        del models_info["model"]

        # Save the numpy array to the file
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        save_file = os.path.join(MODELS_DIR, file_name)

        models_info.to_parquet(save_file) #TODO: rank to meta data of parquet, metadata={"rank": TP['rank']})
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