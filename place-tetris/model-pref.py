import matplotlib.pyplot as plt
import numpy as np
import time

file = './place-tetris/models/models_12x8.npy'

# Prepare the plot outside the function to avoid creating new windows
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(12, 8))
lines = {
    "best_score": ax.plot([], [], label='Best Score', linestyle='-', color='green')[0],
    "average_top5": ax.plot([], [], label='Average Score of Top 5 Models', linestyle='-', color='blue')[0],
    "best_score_in_gen": ax.plot([], [], label='Best Score In Gen', linestyle='-', color='orange')[0],
    "average_score_in_gen": ax.plot([], [], label='Average Score in Gen', linestyle='-', color='red')[0]
}

ax.set_title('Model Performance Over Generations')
ax.set_xlabel('Generation')
ax.set_ylabel('Score')
ax.legend()
ax.grid(True)

def plot_model_performance(file, ax, lines):
    data = np.load(file, allow_pickle=True)[:, 1::-1]
    data = data[data[:, 0].argsort()]

    generations = np.unique(data[:, 0]).astype(int).tolist()
    i_s = np.array([0])
    init = np.vstack((i_s, np.zeros((len(generations), 1))))

    best_score = init.copy()
    average_score_of_top5 = init.copy()
    best_score_in_gen = init.copy()
    average_score_in_gen = init.copy()

    for gen in generations:
        gen_scores = data[data[:, 0] == gen][:, 1]

        all_scores = data[data[:, 0] <= gen][:, 1]
        all_scores = np.sort(all_scores)[::-1]

        best_score_in_gen[gen] = np.max(gen_scores)
        average_score_in_gen[gen] = np.mean(gen_scores)
        best_score[gen] = np.max(best_score_in_gen)
        average_score_of_top5[gen] = np.mean(all_scores[:5])

    generations = [0] + generations

    # Update the data for each plot line
    lines["best_score"].set_data(generations, best_score.ravel())
    lines["average_top5"].set_data(generations, average_score_of_top5.ravel())
    lines["best_score_in_gen"].set_data(generations, best_score_in_gen.ravel())
    lines["average_score_in_gen"].set_data(generations, average_score_in_gen.ravel())

    # Adjust the x-axis and y-axis limits
    ax.set_xlim(0, max(generations))
    ax.set_ylim(0, np.max([best_score, average_score_of_top5, best_score_in_gen, average_score_in_gen]) * 1.1)

    # Redraw the plot
    plt.draw()
    plt.pause(0.1)  # Pause to ensure the plot updates

if __name__ == "__main__":
    while True:
        plot_model_performance(file, ax, lines)
        plt.pause(30)  # Pause for 30 seconds before updating again
