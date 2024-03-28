from place_tetris import TetrisApp, COLS, ROWS
from ai import Model, playMore, expectedScore
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
    "feature_transform": "x,1/(x+0.1),np.ones_like(x)",
    "max_plays": 30
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
t_score = data.sort_values(by="score", ascending=False)["score"].iloc[0]

model = Model(tp, weights.reshape(NUM_EVALS, int(len(weights)/NUM_EVALS)), sigmas, 1)

print(weights.reshape(nft, NUM_EVALS).T)
if "gauss" in ft:
    print(sigmas)

game_params = {
    "gui": False,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 0.01
}

#score, _, _ = model.play(game_params, (0,0), tp)

PLAY_ONCE = False

if PLAY_ONCE:
    score, _, _ = model.play(game_params, (0,0), tp)
    print(f"Score: {score}")
    exit()

print(f"{'Score':<10} {'Mean':<10} {'Exp':<10} {'Std':<10}")
scores = np.zeros(tp["max_plays"])
for i in range(tp["max_plays"]):
    score, _, _ = model.play(game_params, (0,0), tp)
    scores[i] = score

    print(f"{score:10.3f}", end=" ")
    print(f"{np.mean(scores[:i+1]):10.3f}", end=" ")
    print(f"{expectedScore(scores[:i+1]):10.3f}", end=" ")
    print(f"{np.std(scores[:i+1]):10.3f}")

    if not playMore(scores[:i+1]):
        break

print(f"Average score: {np.mean(scores)}")
print(f"Trimmed mean score: {expectedScore(scores)}")
print(f"Stdev score: {np.std(scores):10.3f}")
print(f"Trained score: {t_score}")