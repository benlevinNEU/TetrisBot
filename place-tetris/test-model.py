from place_tetris import TetrisApp, COLS, ROWS
from ai import Model
import numpy as np
from evals import *
import transform_encode as te
import os
import pandas as pd

# Initialize the game
game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 0.01
}

tp = {
    "feature_transform": "self.gauss(x),x,np.ones_like(x)",
}

#weights = np.array([3.9304998446226453, 0.6278212056361121, 30.518039583961556, 34.5815676211182, 27.326096925153074, -3.208231649499382])
ft = tp["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{game_params['rows']}x{game_params['cols']}_{te.encode(ft)}.parquet"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")
models_data_file = os.path.join(MODELS_DIR, file_name)

data = pd.read_parquet(models_data_file)
weights = data.sort_values(by="score", ascending=False)["weights"].iloc[0]
sigmas = data.sort_values(by="score", ascending=False)["sigmas"].iloc[0]

model = Model(tp, weights, sigmas, 1)

print(weights.reshape(nft, NUM_EVALS).T)
print(sigmas)

game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 10,
    "rows": 22,
    "window_pos": (0, 0),
    "sleep": 0.01
}

score, _, _ = model.play(game_params, (0,0), tp)
print(score)