import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import os
import transform_encode as te
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate

# Get correct path to the models data file for current local params setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "../models/")

from local_params import GP, TP

ft = TP["feature_transform"] # Number of feature transforms
#file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.npy"
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"

# Define the rankify function
def rankify(shapes, scales):
    # Calculate the expected value for each distribution in the series
    #expected_values = [np.exp(exp + (std**2) / 2) for exp, std in zip(exp_series, std_series)]
    expected_values = np.exp(np.log(scales) - shapes/2)

    expected_values[np.isnan(expected_values)] = 0
    
    return expected_values

def weighted_integral(exp, std, x_range):
    # Define the distribution
    distribution = stats.lognorm(s=std, scale=np.exp(exp))
    
    # Calculate the probability density over the specified range
    y = distribution.pdf(x_range)
    
    # Calculate the weighted integral
    return integrate.simps(x_range * y, x_range)

# Open all files in MODELS_DIR and search for ones that have file_name[:-8] in them
matching_files = [f for f in os.listdir(MODELS_DIR) if file_name[:-8] in f]

# Import the matching files as pd.DataFrames and merge them into one
merged_df = pd.DataFrame()
for file in matching_files:
    df = pd.read_parquet(os.path.join(MODELS_DIR, file))
    merged_df = pd.concat([merged_df, df])

# Check if a value is an np.ndarray and replace it with a placeholder value
merged_df['exp_score'] = merged_df['exp_score'].apply(lambda x: -1 if isinstance(x, np.ndarray) else x)
merged_df['std'] = merged_df['std'].apply(lambda x: -1 if isinstance(x, np.ndarray) else x)

# Recalculate the "rank" column using the "exp" and "std" columns
merged_df['rank'] = rankify(merged_df['shape'], merged_df['scale'])

# Remove duplicate rows in the merged_df
merged_df = merged_df.drop_duplicates(subset=['exp_score', 'std', 'gen'], keep='first')

merged_df = merged_df.sort_values('rank', ascending=False)

# Save the updated DataFrame to a new parquet file
merged_df.to_parquet(MODELS_DIR + file_name[:-8] + '' + file_name[-8:])





