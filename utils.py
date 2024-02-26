import pstats
import os
import re
import torch
import sys

from heapq import nlargest

def get_newest_generation_number(networks_dir):
    # Compile a regex pattern to match the generation directories and capture their numeric values
    generation_pattern = re.compile(r'generation_(\d+)')
    
    # Get all items in the networks directory
    try:
        all_items = os.listdir(networks_dir)
    except FileNotFoundError:
        os.makedirs(networks_dir, exist_ok=True)
    
    # Filter out directories that match the generation pattern
    try:
        generation_dirs = [item for item in all_items if os.path.isdir(os.path.join(networks_dir, item)) and generation_pattern.match(item)]
    except FileNotFoundError:
        generation_dirs = []
    
    # Extract the generation numbers from the directory names
    generation_numbers = [int(generation_pattern.search(dir_name).group(1)) for dir_name in generation_dirs]
    
    # Find the maximum generation number
    if generation_numbers:
        newest_generation = max(generation_numbers)
    else:
        # Return a default value or raise an error if no generation directories are found
        newest_generation = -1  # Or consider raising an error/exception
    
    return newest_generation

def save_networks(dir, population_score, generation_number):
    # Create a directory for the current generation
    generation_dir = f'{dir}/generation_{generation_number}'
    os.makedirs(generation_dir, exist_ok=True)

    networks = [ps[0] for ps in population_score]
    scores = [ps[1] for ps in population_score]
    
    # Save each network in the population
    for index, (network, score) in enumerate(zip(networks, scores)):
        filename = os.path.join(generation_dir, f'network_{score}.pth')
        torch.save(network.state_dict(), filename)

def find_top_networks(networks_dir, M):
    """
    Find the M networks with the highest scores.

    Args:
        networks_dir (str): Path to the directory containing generation subdirectories.
        M (int): Number of top networks to find.

    Returns:
        list of tuples: List of tuples containing the score and file path of the top M networks.
    """
    # Compile a regular expression to extract the score from the filenames
    filename_pattern = re.compile(r'^\./step-tetris/networks/generation_\d+/network_(\d+\.?\d*)\.pth$')


    # Initialize a list to keep track of the top networks
    top_networks = []

    # Walk through the networks directory and its subdirectories
    for root, dirs, files in os.walk(networks_dir):
        for file in files:
            match = filename_pattern.match(file)
            if match:
                # Extract the score from the filename
                score = float(match.group(1))
                # Append the score and the file path to the top_networks list
                top_networks.append((score, os.path.join(root, file)))

    # Find the M highest scores and their corresponding file paths
    top_M_networks = nlargest(M, top_networks, key=lambda x: x[0])

    return top_M_networks

def get_python_files(directory):
    """Recursively collect all Python files in the given directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        
        for dir in dirs:
            if 'archive' in dir:
                dirs.remove(dir)

        for file in files:
            if file.endswith('.py'):
                # Store the absolute path
                python_files.append(os.path.abspath(os.path.join(root, file)))
    return python_files

def filter_methods(p, directory):
    # Collect all Python files in the directory and its subdirectories
    valid_paths = get_python_files(directory)

    filtered_stats = {}

    # Filter the stats
    for func in p.stats:
        # Check if the function's file path is in the list of valid paths
        if func[0] in valid_paths:
            filtered_stats[func] = p.stats[func]

    # Create a new Stats object with filtered stats
    new_p = pstats.Stats()
    new_p.stats = filtered_stats
    new_p.total_tt = p.total_tt  # Copy total time as well, important for some displays

    return new_p

def merge_profile_stats(directory):
    directory = os.path.abspath(directory)
    profile_files = [f for f in os.listdir(directory) if f.endswith('.prof')]

    # Initialize a dictionary to accumulate stats
    merged_stats = {}

    for filename in profile_files:
        path = os.path.join(directory, filename)
        current_stats = pstats.Stats(path)

        for func, (cc, nc, tt, ct, callers) in current_stats.stats.items():
            
            if func not in merged_stats:
                merged_stats[func] = [cc, nc, tt, ct, callers.copy()]
            else:
                merged_entry = merged_stats[func]
                # Accumulate calls, total and cumulative times
                merged_entry[0] += cc  # cumulative calls
                merged_entry[1] += nc  # ncalls
                merged_entry[2] += tt  # total time
                merged_entry[3] += ct  # cumulative time

                # Merge callers information
                for caller, caller_stats in callers.items():
                    if caller in merged_entry[4]:
                        merged_entry[4][caller] = tuple(map(sum, zip(merged_entry[4][caller], caller_stats)))
                    else:
                        merged_entry[4][caller] = caller_stats

    # Create a Stats object with the merged stats
    merged_pstats = pstats.Stats()
    merged_pstats.stats = merged_stats
    merged_pstats.total_calls = sum(entry[0] for entry in merged_stats.values())
    merged_pstats.total_tt = sum(entry[2] for entry in merged_stats.values())

    # You can use merged_pstats just like any Stats object
    #merged_pstats#.print_stats(30)

    return merged_pstats

def print_to_line(lock, string, line):
    with lock:
        sys.stdout.write(f'\033[{line}B') # Move cursor dow
        sys.stdout.flush()
        sys.stdout.write(string)
        sys.stdout.flush()  # Ensure 'string' is processed
        sys.stdout.write(f'\033[{line+1}A')  # Move cursor up
        sys.stdout.flush()  # Ensure cursor move is processed
