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
    "feature_transform": "x,1/(x+0.1),np.ones_like(x)",
    "plays": 20
}

#weights = np.array([3.9304998446226453, 0.6278212056361121, 30.518039583961556, 34.5815676211182, 27.326096925153074, -3.208231649499382])
ft = tp["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{game_params['rows']}x{game_params['cols']}_{te.encode(ft)}_{tp['plays']}.parquet"

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

games = 30

def playMore(scores, threshold=0.0075, max_count=games):

    if len(scores) < 3:
        return max_count  # Not enough data to make a decision

    new_std = np.std(scores)
    prev_std = np.std(scores[:-1])
    if abs(new_std - prev_std) / prev_std < threshold:
        return False  # The number of games where the estimate stabilized

    return True  # Return the max games if the threshold is never met

from scipy import stats

def trimProp(scores):
    std = np.std(scores)
    samples = len(scores)

    # Base prop
    base_prop = 0.2

    # Adjust the proportion based on the standard deviation and sample size
    # This is a simple heuristic and can be adjusted based on empirical testing
    prop = base_prop * std / np.sqrt(samples)

    # Ensure the proportion is within a sensible range, e.g., 0.01 to 0.25
    prop = max(0.01, min(prop, 0.25))

    return np.mean(stats.mstats.winsorize(scores, limits=(prop, prop)))

scores = np.zeros(games)

print(f"{'Score':<10} {'Mean':<10} {'Exp':<10} {'Std':<10}")

for i in range(games):
    score, _, _ = model.play(game_params, (0,0), tp)
    scores[i] = score

    print(f"{score:10.3f}", end=" ")
    print(f"{np.mean(scores[:i+1]):10.3f}", end=" ")
    print(f"{trimProp(scores[:i+1]):10.3f}", end=" ")
    print(f"{np.std(scores[:i+1]):10.3f}")

    if not playMore(scores[:i+1]):
        break
        print("Can stop playing @ game", i)
        pass

print(f"Average score: {np.mean(scores)}")
print(f"Trimmed mean score: {trimProp(scores)}")
print(f"Stdev score: {np.std(scores):10.3f}")
print(f"Trained score: {t_score}")