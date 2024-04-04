from place_tetris import TetrisApp, COLS, ROWS
from ai import Model, playMore, expectedScore
import numpy as np
from evals import *
import transform_encode as te
import os, cProfile
import pandas as pd
from scipy.stats import norm, lognorm, gamma, weibull_min
import time, utils
from get_latest_profiler_data import print_stats

# Initialize the game
game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 10,
    "rows": 20,
    "window_pos": (0, 0),
    "sleep": 0.1
}

tp = {
    "feature_transform": "x",
    "max_plays": 30,
    "rank": lambda e,s: e - np.sqrt(s),
    "profile": True
}

def playMore(scores, threshold=0.04, min_count=8, max_count=tp["max_plays"]):

    if len(scores) < min_count:
        return max_count  # Not enough data to make a decision

    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores, floc=0)
    log_likelihood_lognorm = np.sum(lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    new_aic = 2*3 - 2*log_likelihood_lognorm

    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores[:-1], floc=0)
    log_likelihood_lognorm = np.sum(lognorm.logpdf(scores[:-1], shape_lognorm, loc_lognorm, scale_lognorm))
    prev_aic = 2*3 - 2*log_likelihood_lognorm

    if abs(new_aic - prev_aic) / prev_aic < threshold:
        return False  # The number of games where the estimate stabilized

    return True  # Return the max games if the threshold is never met

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

#data = pd.read_parquet(models_data_file)
#weights = data.sort_values(by="rank", ascending=False)["weights"].iloc[0]
#sigmas = data.sort_values(by="rank", ascending=False)["sigmas"].iloc[0]
#t_score = data.sort_values(by="rank", ascending=False)["exp_score"].iloc[0]
#t_std = data.sort_values(by="rank", ascending=False)["std"].iloc[0]
#t_rank = data.sort_values(by="rank", ascending=False)["rank"].iloc[0]

weights = np.array([3.9304998446226453, 0.6278212056361121, 30.270904991356566, 34.65472026088795, 27.326096925153074, -3.208231649499382, 0, 0, 0])
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
'''game_params = {
    "gui": False,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 8,
    "rows": 12,
    "window_pos": (0, 0),
    "sleep": 0.01
}'''

model = Model(tp, weights.reshape(NUM_EVALS, int(len(weights)/NUM_EVALS)), sigmas, 1)
model.evals = Evals(game_params)

#score, _, _ = model.play(game_params, (0,0), tp)

PLAY_ONCE = True

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
iters = 15
for iter in range(iters):
    print(f"Iteration: {iter+1} / {iters}")
    print(f"{'Score':^{di}} {'Mean':^{di}} {'Exp':^{di}} {'Aic':^{di}} {'Dev':^{di}}")
    scores = np.ones(tp["max_plays"])
    for i in range(tp["max_plays"]):
        score, _, _, _ = model.play(game_params, (0,0), tp, ft)
        scores[i] = score

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
        print(f"{dev:{di-1}.2f}%")

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
    print(f"Average score: {np.mean(scores):{sp}.1f}")
    print(f"{' ':<6} {'Measured':^{sp}} {'Expected':^{sp}}")
    print(f"{'Exp':<6} {f'{expectedScore(scores):.1f}':>{sp}} {f'{t_score:.1f}':>{sp}}")
    print(f"{'Aic':<6} {f'{aic(scores):.1f}':>{sp}} {f'{t_std:.1f}':>{sp}}")
    rank = rank(shape_lognorm, scale_lognorm)
    print(f"{'Rank':<6} {f'{rank:.1f}':>{sp}} {f'{t_rank:.1f}':>{sp}}\n")

    data.append(scores)

import matplotlib.pyplot as plt

# Save the figure
plt.figure(figsize=(20, 15))
i=0
agr_data = pd.DataFrame(columns=["norm", "log-norm", "gamma", "weibull"])
for i in range(iters):
    scores = data[i]
    n = len(scores)

    plt.subplot(5, 5, i+1)
    plt.hist(scores, bins=30, edgecolor='black', density=True)
    plt.title(f'Iteration {i+1}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    x = np.linspace(np.min(scores), np.max(scores), 100)

    # Normal distribution
    mu, sigma = norm.fit(scores)
    p_norm = norm.pdf(x, mu, sigma)
    med_norm = norm.ppf(0.5, mu, sigma)
    # Grade the normal distribution
    log_likelihood_norm = np.sum(norm.logpdf(scores, mu, sigma))
    aic_norm = 2*2 - 2*log_likelihood_norm
    bic_norm = np.log(n)*2 - 2*log_likelihood_norm

    # Log-normal distribution
    shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(scores, floc=0)
    p_lognorm = lognorm.pdf(x, shape_lognorm, loc_lognorm, scale_lognorm)
    #med_lognorm = lognorm.ppf(0.5, shape_lognorm, loc_lognorm, scale_lognorm)
    med_lognorm = scale_lognorm
    # Grade the log-normal distribution
    log_likelihood_lognorm = np.sum(lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    aic_lognorm = 2*3 - 2*log_likelihood_lognorm
    bic_lognorm = np.log(n)*3 - 2*log_likelihood_lognorm

    # Gamma distribution
    alpha_gamma, loc_gamma, beta_gamma = gamma.fit(scores, floc=0)
    p_gamma = gamma.pdf(x, alpha_gamma, loc_gamma, beta_gamma)
    med_gamma = gamma.ppf(0.5, alpha_gamma, loc_gamma, beta_gamma)
    # Grade the gamma distribution
    log_likelihood_gamma = np.sum(gamma.logpdf(scores, alpha_gamma, loc_gamma, beta_gamma))
    aic_gamma = 2*3 - 2*log_likelihood_gamma
    bic_gamma = np.log(n)*3 - 2*log_likelihood_gamma

    # Weibull distribution
    shape_weibull, loc_weibull, scale_weibull = weibull_min.fit(scores, floc=0)
    p_weibull = weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull)
    med_weibull = weibull_min.ppf(0.5, shape_weibull, loc_weibull, scale_weibull)
    # Grade the weibull distribution
    log_likelihood_weibull = np.sum(weibull_min.logpdf(scores, shape_weibull, loc_weibull, scale_weibull))
    aic_weibull = 2*3 - 2*log_likelihood_weibull
    bic_weibull = np.log(n)*3 - 2*log_likelihood_weibull

    plt.plot(x, p_norm, 'r-', label=f'Norm: {med_norm:.1f} - ({aic_norm:.1f}, {bic_norm:.1f})')
    plt.plot(x, p_lognorm, 'g-', label=f'Log-Norm: {med_lognorm:.1f} - ({aic_lognorm:.1f}, {bic_lognorm:.1f})')
    plt.plot(x, p_gamma, 'b-', label=f'Gamma: {med_gamma:.1f} - ({aic_gamma:.1f}, {bic_gamma:.1f})')
    plt.plot(x, p_weibull, '-', color='orange', label=f'Weibull: {med_weibull:.1f} - ({aic_weibull:.1f}, {bic_weibull:.1f})')
    plt.title(f'Distribution of Scores: Exp = {expectedScore(scores):.1f}')
    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.legend()

    df = pd.DataFrame({"norm": [med_norm], "log-norm": [med_lognorm], "gamma": [med_gamma], "weibull": [med_weibull]})
    agr_data = pd.concat([agr_data, df], ignore_index=True)

print("Aggregated Data")
print(f"{' ':<10} {'Mean':<10} {'Std':<10}")
print(f"{'Norm':<10} {agr_data['norm'].mean():<10.2f} {agr_data['norm'].std():<10.2f}")
print(f"{'Log-Norm':<10} {agr_data['log-norm'].mean():<10.2f} {agr_data['log-norm'].std():<10.2f}")
print(f"{'Gamma':<10} {agr_data['gamma'].mean():<10.2f} {agr_data['gamma'].std():<10.2f}")
print(f"{'Weibull':<10} {agr_data['weibull'].mean():<10.2f} {agr_data['weibull'].std():<10.2f}")

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_DIR, "figure.png"), bbox_inches='tight')
plt.close()

if PROFILE:
    profiler.disable()
    profiler.dump_stats(f"{PROF_DIR}{tid}/main.prof")

    p = utils.merge_profile_stats(profiler_dir)
    print_stats(utils.filter_methods(p, CURRENT_DIR).strip_dirs().sort_stats('tottime'))
    print_stats(p.strip_dirs().sort_stats('tottime'), 30)