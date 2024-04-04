
import os
import pandas as pd
import transform_encode as te
from local_params import GP, TP

ft = TP["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")
models_data_file = os.path.join(MODELS_DIR, file_name)

data = pd.read_parquet(models_data_file)

print(data[:TP['top_n']][['rank', 'exp_score', 'std', 'gen']])

