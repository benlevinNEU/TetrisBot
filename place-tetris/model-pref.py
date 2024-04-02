import matplotlib.pyplot as plt
import numpy as np
import time, os
import transform_encode as te
import pandas as pd

# Get correct path to the models data file for current local params setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")
from local_params import GP, TP
ft = TP["feature_transform"]
nft = ft.count(',') + 1
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
models_data_file = os.path.join(MODELS_DIR, file_name)

topN = TP["top_n"]

plt.style.use('dark_background')

def init_plot():
    # Prepare the plot outside the function to avoid creating new windows
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    lines = {
        "best_rank": ax.plot([], [], label='Best Score', linestyle='-', color='green')[0],
        "average_top5": ax.plot([], [], label=f'Average Score of Top {topN} Models', linestyle='-', color='blue')[0],
        "best_rank_in_gen": ax.plot([], [], label='Best Score In Gen', linestyle='-', color='orange')[0],
        "average_rank_in_gen": ax.plot([], [], label='Average Score in Gen', linestyle='-', color='red')[0]
    }

    ax.set_title('Model Performance Over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Rank')
    ax.legend()
    ax.grid(True)

    return fig, ax, lines

def plot_model_performance(file, ax, lines):
    while True:
        try:
            data = pd.read_parquet(file)
            break
        except:
            time.sleep(1)
    data = data.sort_values(by="rank", ascending=False)
    data = data[["gen", "rank"]].values

    generations = int(np.max(data[:, 0])) + 1
    i_s = np.array([0])
    init = np.vstack((i_s, np.zeros((generations, 1))))

    best_score = init.copy()
    average_score_of_topN = init.copy()
    best_score_in_gen = init.copy()
    average_score_in_gen = init.copy()

    for gen in range(generations):
        gen_scores = data[data[:, 0] == gen][:, 1]

        all_scores = data[data[:, 0] <= gen][:, 1]
        all_scores = np.sort(all_scores)[::-1]

        best_score_in_gen[gen] = np.max(gen_scores)
        average_score_in_gen[gen] = np.mean(gen_scores)
        best_score[gen] = np.max(best_score_in_gen)
        average_score_of_topN[gen] = np.mean(all_scores[:topN])


    # Update the data for each plot line
    x = np.linspace(0,generations-1,generations)
    lines["best_rank"].set_data(x, best_score.ravel()[:-1])
    lines["average_top5"].set_data(x, average_score_of_topN.ravel()[:-1])
    lines["best_rank_in_gen"].set_data(x, best_score_in_gen.ravel()[:-1])
    lines["average_rank_in_gen"].set_data(x, average_score_in_gen.ravel()[:-1])

    # Adjust the x-axis and y-axis limits
    ax.set_xlim(0, generations-1)
    ax.set_ylim(0, np.max([best_score, average_score_of_topN, best_score_in_gen, average_score_in_gen]) * 1.1)

    # Redraw the plot
    plt.draw()
    plt.pause(0.1)  # Pause to ensure the plot updates

if __name__ == "__main__":

    if not os.path.exists(models_data_file):
        print(f"The following file does not exist: \n{models_data_file}")
        exit(1)

    fig, ax, lines = init_plot()

    while True:
        plot_model_performance(models_data_file, ax, lines)
        plt.pause(30)  # Pause for 30 seconds before updating again

        if not plt.fignum_exists(fig.number):
            break