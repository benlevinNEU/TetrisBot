from place_tetris import TetrisApp, COLS, ROWS
from ai import Model, playMore, expectedScore
import numpy as np
from evals import *
import transform_encode as te
import os, cProfile, multiprocessing
import pandas as pd
from scipy.stats import norm, lognorm, gamma, weibull_min
import time, utils
from get_latest_profiler_data import print_stats

# Initialize the game
game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 10,
    "rows": 14,
    "window_pos": (0, 0),
    "sleep": 0.1
}

tp = {
    "feature_transform": "x",
    "max_plays": 20,
    "profile": False,
    "prune_ratio": 0.3,
    "cutoff": 300,
    "demo": False,
    "workers": 4
}

MAX_WORKERS = tp["workers"] if tp["workers"] > 0 else multiprocessing.cpu_count()

#weights = np.array([3.9304998446226453, 0.6278212056361121, 30.518039583961556, 34.5815676211182, 27.326096925153074, -3.208231649499382])
ft = tp["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{game_params['rows']}x{game_params['cols']}_{te.encode(ft)}.parquet"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")
models_data_file = os.path.join(MODELS_DIR, file_name)

PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
PROFILE = tp["profile"]

tid = int(time.time())
profiler_dir = f"{PROF_DIR}{tid}/"

if PROFILE:
    os.makedirs(profiler_dir)
    profiler = cProfile.Profile()
    profiler.enable()

'''data = pd.read_parquet(models_data_file)
weights = data.sort_values(by="rank", ascending=False)["weights"].iloc[0]
sigmas = data.sort_values(by="rank", ascending=False)["sigmas"].iloc[0]
t_score = data.sort_values(by="rank", ascending=False)["exp_score"].iloc[0]
t_std = data.sort_values(by="rank", ascending=False)["std"].iloc[0]
t_rank = data.sort_values(by="rank", ascending=False)["rank"].iloc[0]'''

weights = np.array([3.74041, 0.43773, 30.08081, 34.46463, 27.13600, -3.39832, -0.19009, -0.19009, -0.19009])
sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0])
t_score = 0
t_std = 0
t_rank = 0

# Format the weights for display with labels
feature_transforms = ft.split(",")
sp = 10
print(f'Labled Training Parameters')
feature_transforms = [val if val != 'np.ones_like(x)' else '1' for val in feature_transforms]
print(f'{' ':<{sp}}' + " ".join([f'{val:^{sp}}' for val in feature_transforms]))
labels = getEvalLabels()
for i, row in enumerate(weights.reshape(NUM_EVALS, nft)):
    label = labels[i]
    print(f"{label:<{sp}}{' '.join([f'{f'{val:.5f}':>{sp}}' for val in row])}")

if "gauss" in ft:
    print(sigmas)

print('\n', end='')

# Uncomment if you want to test a model trained on a different board size
game_params = {
    "gui": False,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 10,
    "rows": 14,
    "window_pos": (0, 0),
    "sleep": 0.1
}

model = Model(tp, weights.reshape(NUM_EVALS, int(len(weights)/NUM_EVALS)), sigmas, 1)
model.evals = Evals(game_params)

PLAY_ONCE = False

ft = eval(f"lambda self, x: np.column_stack([{te.decode(tp["feature_transform"])}])")
if PLAY_ONCE:
    score, _, _ = model.play(game_params, (0,0), tp, ft)
    print(f"Score: {score}")

    if PROFILE:
        profiler.disable()
        profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

        p = utils.merge_profile_stats(profiler_dir)
        print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
        print_stats(p.strip_dirs().sort_stats('tottime'), 30)
    exit()

def aic(scores):
    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores, floc=0)
    log_likelihood_lognorm = np.sum(lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    return  2*3 - 2*log_likelihood_lognorm

di = 8
data = []
iters = 4

import sys

def runUntilConverge(it):
    it = it[0] + 1

    #print(f"Iteration: {iter+1} / {iters}")
    #print(f"{'Score':^{di}} {'Mean':^{di}} {'Exp':^{di}} {'Aic':^{di}} {'Dev':^{di}} {'Time':^{di}}")
    scores = np.ones(tp["max_plays"])
    for i in range(tp["max_plays"]):
        start = time.time()
        score, _, _ = model.play(game_params, (0,0), tp, ft)
        end = time.time()
        scores[i] = score
        
        desc = f'{it}: {i+1}/{tp["max_plays"]}'
        print(f"{desc:<{di}}", end=" ")
        print(f"{score:{di}.1f}", end=" ")
        print(f"{np.mean(scores[:i+1]):{di}.1f}", end=" ")
        print(f"{expectedScore(scores[:i+1]):{di}.1f}", end=" ")
        print(f"{aic(scores[:i+1]):{di}.1f}", end=" ")

        # Calculate the deviation between stdevs
        if i>1: 
            prev = aic(scores[:i])
            dev = (aic(scores[:i+1]) - prev) / (prev) * 100
        else: 
            dev = np.inf
        print(f"{dev:{di-1}.2f}%", end=" ")

        print(f"{(end-start):{di}.2f}s")

        if not playMore(scores[:i+1]):
            break

    scores = scores[:i+1]

    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores, floc=0)
    def rank(shape, scale):
        exp = np.log(scale)
        std = shape
        expected_value = np.exp(exp - std/2)

        return expected_value

    sp = 10
    #print(f"Average score: {np.mean(scores):{sp}.1f}")
    #print(f"{' ':<6} {'Measured':^{sp}} {'Expected':^{sp}}")
    #print(f"{'Exp':<6} {f'{expectedScore(scores):.1f}':>{sp}} {f'{t_score:.1f}':>{sp}}")
    #print(f"{'Aic':<6} {f'{aic(scores):.1f}':>{sp}} {f'{t_std:.1f}':>{sp}}")
    #rank = rank(shape_lognorm, scale_lognorm)
    #print(f"{'Rank':<6} {f'{rank:.1f}':>{sp}} {f'{t_rank:.1f}':>{sp}}\n")

    return scores

import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    print(f"{'Itter':^{di}} {'Score':^{di}} {'Mean':^{di}} {'Exp':^{di}} {'Aic':^{di}} {'Dev':^{di}} {'Time':^{di}}")
    
    futures = [executor.submit(runUntilConverge, (i, )) for i in range(iters)]
    for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
        sys.stdout.write(f"Completed {count}/{iters}    \r")
        sys.stdout.flush()
        data.append(future.result())

import matplotlib.pyplot as plt

# Save the figure
plt.figure(figsize=(10, 10))
i=0
for i in range(iters):
    scores = data[i]
    n = len(scores)

    plt.subplot(2, 2, i+1)
    plt.hist(scores, bins=30, edgecolor='black', density=True)
    plt.title(f'Iteration {i+1}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    x = np.linspace(np.min(scores), np.max(scores), 100)

    # Log-normal distribution
    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores, floc=0)
    p_lognorm = lognorm.pdf(x, shape_lognorm, loc_lognorm, scale_lognorm)
    #med_lognorm = lognorm.ppf(0.5, shape_lognorm, loc_lognorm, scale_lognorm)
    med_lognorm = scale_lognorm
    # Grade the log-normal distribution
    log_likelihood_lognorm = np.sum(lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    aic_lognorm = 2*3 - 2*log_likelihood_lognorm
    bic_lognorm = np.log(n)*3 - 2*log_likelihood_lognorm

    plt.plot(x, p_lognorm, 'g-', label=f'Log-Norm: {med_lognorm:.1f} - ({aic_lognorm:.1f})')
    plt.title(f'Distribution of Scores: Exp = {expectedScore(scores):.1f}')
    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_DIR, "figure-20x10.png"), bbox_inches='tight')
plt.close()

if PROFILE:
    profiler.disable()
    profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

    p = utils.merge_profile_stats(profiler_dir)
    print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
    print_stats(p.strip_dirs().sort_stats('tottime'), 30)