import os
import re
import matplotlib.pyplot as plt
import numpy as np

file ='./place-tetris/models/models_12x8.npy'

data = np.load(file, allow_pickle=True)[:, 1::-1]
data = data[data[:, 0].argsort()]

generations = np.unique(data[:, 0]).astype(int).tolist()
i_s = np.array([1000])
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
    average_score_of_top5[gen] = np.mean(np.sort(all_scores)[-5:])

'''
best_score = np.maximum.accumulate(np.array([np.max(data[data[:, 0] == gen, 1]) for gen in generations])).tolist()
average_score_of_top5 = np.zeros((len(data), 2)).tolist()
best_score_in_gen = [np.max(group[:, 1]) for group in np.split(data[:, 1], idx[1:])]
average_score_in_gen = np.zeros((len(data), 2)).tolist()
'''

generations = [0] + generations

plt.figure(figsize=(12, 8))

plt.plot(generations, best_score.tolist(), label='Best Score', linestyle='-', color='green')
plt.plot(generations, average_score_of_top5.tolist(), label='Average Score of Top 5 Models', linestyle='-', color='blue')
plt.plot(generations, best_score_in_gen.tolist(), label='Best Score In Gen', linestyle='-', color='orange')
plt.plot(generations, average_score_in_gen.tolist(), label='Average Score in Gen', linestyle='-', color='red')

plt.title('Model Performance Over Generations')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
