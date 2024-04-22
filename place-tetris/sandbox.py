import numpy as np
import pandas as pd
import os
import transform_encode as te
from evals import NUM_EVALS
from local_params import GP, TP

# Get correct path to the models data file for current local params setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

ft = TP["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
models_data_file = os.path.join(MODELS_DIR, file_name)

data = pd.read_parquet(models_data_file).sort_values(by="rank", ascending=False)
print(data[:TP["top_n"]][["rank", "exp_score", "std", "gen"]], end='\n\n')

exp_data = data.sort_values(by="exp_score", ascending=False)
print(exp_data[:TP["top_n"]][["rank", "exp_score", "std", "gen"]], end='\n\n')

print('Best model:')
print(data.head(5)["weights"].values[0].reshape(NUM_EVALS, nft))

#ft = "x,x**2"
#file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
#models_data_file = os.path.join(MODELS_DIR, file_name)

# Add NUM_EVALS 0's to each array in the 'weights' column
#NUM_EVALS = 9
#data['sigmas'] = data['sigmas'].apply(lambda x: np.full(NUM_EVALS,np.nan))
#data['s_cost_metrics'] = data['s_cost_metrics'].apply(lambda x: np.full(NUM_EVALS,np.nan))

# Print a single row of data with all values
new_data = data.sort_values(by="gen", ascending=False)
print(new_data[:TP["top_n"]][["rank", "exp_score", "std", "gen"]], end='\n\n')
print(new_data.iloc[0]['weights'])

#data.to_parquet(models_data_file)