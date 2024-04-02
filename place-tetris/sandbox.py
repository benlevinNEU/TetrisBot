import numpy as np
import pandas as pd
import os
import transform_encode as te

# Get correct path to the models data file for current local params setup
'''CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")
from local_params import GP, TP
ft = TP["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
models_data_file = os.path.join(MODELS_DIR, file_name)

data = pd.read_parquet(models_data_file)
print(data.head())
print(data.head(1)["weights"].values)'''