import os
import re
import matplotlib.pyplot as plt
import numpy as np

networks_dir = './frame-tetris/networks'
generation_pattern = re.compile(r'generation_(\d+)')
score_pattern = re.compile(r'network_(\d+)\.pth')

average_scores = []
best_scores_per_generation = []
top_10_averages = []
all_scores = []

for root, dirs, files in os.walk(networks_dir):
    generation_match = generation_pattern.search(root)
    if generation_match:
        generation_scores = []
        for file in files:
            score_match = score_pattern.search(file)
            if score_match:
                score = float(score_match.group(1))
                generation_scores.append(score)
                all_scores.append(score)
        
        if generation_scores:
            average_score = sum(generation_scores) / len(generation_scores)
            average_scores.append(average_score)
            best_generation_score = max(generation_scores)
            best_scores_per_generation.append(best_generation_score)

        sorted_all_scores = sorted(all_scores, reverse=True)[:10]
        if len(sorted_all_scores) > 0:
            top_10_averages.append(np.mean(sorted_all_scores))

generations = list(range(1, len(average_scores) + 1))

plt.figure(figsize=(12, 8))

plt.plot(generations, average_scores, label='Average Score per Generation')
plt.plot(generations, best_scores_per_generation, label='Best Score per Generation')
plt.plot(generations, np.maximum.accumulate(best_scores_per_generation), label='Best Cumulative Score', linestyle='--')

if len(average_scores) != len(top_10_averages):
    # If there's a mismatch in length, align top_10_averages with the number of generations processed
    top_10_averages = top_10_averages[:len(average_scores)]

plt.plot(generations, top_10_averages, label='Average of Top 10 Scores Over Time', linestyle='--', color='red')

plt.title('Network Performance Over Generations')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
