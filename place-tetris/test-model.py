from place_tetris import TetrisApp, COLS, ROWS
from ai import Model
import numpy as np

# Parameters for the evolutionary process
population_size = 60
top_n = 5
generations = 1000
plays = 2  # Number of games to play for each model each generation

# Initialize the game
game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 20,
    "cols": 10,
    "rows": 22,
    "window_pos": (0, 0)
}

weights = np.array([3.9304998446226453, 0.6278212056361121, 30.518039583961556, 34.5815676211182, 27.326096925153074, -3.208231649499382])

model = Model(weights)
score = model.play(game_params, (0,0))
print(score)