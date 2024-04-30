import numpy as np
import pandas as pd
import os, sys, time, cProfile, multiprocessing, platform, re
from multiprocessing import Manager, Pool, Value
import concurrent.futures
import utils, importlib

from get_latest_profiler_data import print_stats
from evals import NUM_EVALS
import transform_encode as te
from model import Model, FT, MAX_WORKERS, CURRENT_DIR

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
    "top_n": 40,
    "generations": 1000,
    "max_plays": 30,
    "mutation_rate": lambda gen: 0.1 * np.exp(-0.01 * gen) + 0.01,
    "mutation_strength": lambda gen: 0.001 * np.exp(-0.02 * gen) + 0.00001,
    "s_mutation_strength": lambda gen: 0.001 * np.exp(-0.02 * gen) + 0.00001,
    "profile": False,
    "workers": 0,
    "feature_transform": "x,x**3,x**(1/3)",
    "learning_rate": lambda gen: 0.01 * np.exp(-0.02 * gen) + 0.0001,
    "s_learning_rate": lambda gen: 0.01 * np.exp(-0.02 * gen) + 0.0001,
    "age_factor": lambda x: 1.2**x,
    "p_random": 0.2,
    "prune_ratio": 0.2,
    "cutoff": 500,
    'lin_prop_max': 0.2,
    "demo": False,
    "snap_prob": lambda mxh: 0.12 * mxh**8,
    "use_snap_prob": 0.5,
}
'''

PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

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

            out = f"\r{prnt_lbl}: {0:.2f}% of {len(args[0])} - Elapsed / Est (s): {0:.0f} / Unknown             " # Overwrites the line
            sys.stdout.write(out)
            sys.stdout.flush()

            progress_log = open(f"{CURRENT_DIR}/progress.log", "a")
            progress_log.seek(0, os.SEEK_END)  # Seek to the end of the file
            progress_log.seek(progress_log.tell() - len(out), os.SEEK_SET)  # Seek to the start of the last line
            progress_log.truncate()  # Truncate the file from this point
            progress_log.write(out)  # Write the new line
            progress_log.close()

            # print(prnt_lbl)
            # Track progress and collect results
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            #for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
                progress = (count / len(args[0])) * 100
                # Prints the progress percentage with appropriate task label
                e_time = time.time() - start_time
                out = f"\r{prnt_lbl}: {progress:.2f}% of {len(args[0])} - Elapsed / Est (s): {e_time:.0f}/{e_time/progress *100:.0f}             " # Overwrites the line
                
                sys.stdout.write(out)
                sys.stdout.flush()

                progress_log = open(f"{CURRENT_DIR}/progress.log", "a")
                progress_log.write(out)
                progress_log.close()
                
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
        else:
            raise ValueError(f"Parent ID {parent_id} not found in models for child ID {child_id}")

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
        out = f"\rGeneration {generation}: Top Expected Score = {top_score:.1f}"
        
        print(out)
        progress_log = open(f"{CURRENT_DIR}/progress.log", "a")
        progress_log.write(out)
        progress_log.close()

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