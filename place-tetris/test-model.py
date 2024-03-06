from place_tetris import TetrisApp, COLS, ROWS
from ai import Model
import numpy as np
from evals import *

# Initialize the game
game_params = {
    "gui": True,  # Set to True to visualize the game
    "cell_size": 30,
    "cols": 10,
    "rows": 22,
    "window_pos": (0, 0),
    "sleep": 0.01
}

tp = {
    "feature_transform": "x",
}

weights = np.array([3.9304998446226453, 0.6278212056361121, 30.518039583961556, 34.5815676211182, 27.326096925153074, -3.208231649499382])

model = Model(tp, weights)
score, _ = model.play(game_params, (0,0), tp)
print(score)