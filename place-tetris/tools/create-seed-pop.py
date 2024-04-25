import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from local_params import GP, TP
import transform_encode as te

# Setup directories and filenames
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "../models/")
ft = TP["feature_transform"]
nft = ft.count(',') + 1  # Assuming this correctly calculates the number of features
source_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}_old.parquet"
sink_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
source_file_path = os.path.join(MODELS_DIR, source_name)
sink_file_path = os.path.join(MODELS_DIR, sink_name)

# Read the data
data = pd.read_parquet(source_file_path)

data['weights'] = data['weights'].astype(object)

# Modify the weights for the first 30 rows (or up to top_n if less than 30)
'''
if isinstance(data.iloc[0]['weights'], np.ndarray):
    data['weights'] = data['weights'].apply(lambda x: x.tolist())
top_n_mod = min(len(data), TP['top_n'])
new_data = pd.DataFrame()
for i in range(top_n_mod):
    weights = np.array(data.iloc[i]['weights'])
    score = data.iloc[i]['score']
    gen = data.iloc[i]['gen']
    # Assuming the weights are already appropriately shaped
    new_weights = weights.reshape(3, 6)
    new_weights = np.hstack((new_weights, np.zeros((3, 3))))  # Add 3 zeros to each row of weights
    blank_cm = np.zeros((19))
    blank_cm[0] = 1
    new_data = pd.concat((new_data, pd.DataFrame({'score': [score], 
                                                  'gen': [gen], 
                                                  'weights': [new_weights.flatten()], 
                                                  'cost_metrics': [blank_cm]})))

# Save the modified data to a new file
new_data.to_parquet(sink_file_path)'''

# Modify weights for every row using apply() method
def modify_weights(row):
    weights = np.array(row['weights']).reshape(3, 6)
    new_weights = np.hstack((weights, np.zeros((weights.shape[0], 3))))
    # Return the modified weights as a list to store it as a single object in DataFrame
    return new_weights.flatten()

def modify_cm(row):
    cm = np.array(row['cost_metrics'][1:]).reshape(2, 6)
    new_weights = np.hstack((cm, np.zeros((cm.shape[0], 3))))
    # Return the modified weights as a list to store it as a single object in DataFrame
    return np.hstack((row['cost_metrics'][1], new_weights.flatten()))

# Apply the function to modify weights
data['weights'] = data.apply(modify_weights, axis=1)
data['cost_metrics'] = data.apply(modify_cm, axis=1)

# Save modified DataFrame to new file
data.to_parquet(sink_file_path)