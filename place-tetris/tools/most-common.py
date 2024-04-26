import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import os
import transform_encode as te

# Get correct path to the models data file for current local params setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

file_name = os.path.join(CURRENT_DIR, "../success.log")

def read_and_group_lines(filename):
    # Open the file and read lines
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Dictionary to store groups of lines
    groups = []

    # Group lines according to specified pattern
    for gn in range(0, len(lines), 10):

        group = lines[gn+1:gn+10]
        # Process each string to remove unwanted characters and convert to float
        str = [s.strip().strip('[]') for s in group]
        cleaned_group = []
        for s in str:
            cleaned_group.extend([f"{float(val):.6f}" for val in s.split()])
        cleaned_group = np.array(cleaned_group)
        
        groups.append(cleaned_group)

    return groups

def find_most_frequent_group(groups):
    from collections import Counter

    # Convert each NumPy array to a tuple for hashing
    group_tuples = [tuple(group.ravel()) for group in groups]  # Use ravel() to flatten the array, ensuring one-dimensional tuples
    group_counts = Counter(group_tuples)
    
    most_common = group_counts.most_common(5)

    return most_common

groups = read_and_group_lines(file_name)
most_common = find_most_frequent_group(groups)

for group, count in most_common:
    print(f"Has succeeded {count} times. Here are all the weights for this model:")
    weights = np.array([float(val) for val in group])
    print(weights.reshape(9, 3))
