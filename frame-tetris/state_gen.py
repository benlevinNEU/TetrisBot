import numpy as np

import os, sys
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tetris_utils import *

def get_stone_states(stone_id):

    shape = np.array(tetris_shapes[stone_id])
    shape = np.where(shape > 0, 1, 0)

    states = []

    for i in range(4):

        board_slice = np.zeros((shape.shape[0], COLS), dtype=int)

        left_shift, right_shift, up_shift, down_shift = getShifts(shape)

        # Find the indices where the trimming should stop
        row_nonzero = np.where(shape.any(axis=1))[0]
        col_nonzero = np.where(shape.any(axis=0))[0]
        
        if row_nonzero.size > 0:
            trimmed_shape = shape[row_nonzero[0]:row_nonzero[-1]+1, :]
        if col_nonzero.size > 0:
            trimmed_shape = trimmed_shape[:, col_nonzero[0]:col_nonzero[-1]+1]

        for x in range(COLS):
            state = np.copy(board_slice)
            if x + trimmed_shape.shape[1] <= COLS:
                state[up_shift:up_shift+trimmed_shape.shape[0], x:x+trimmed_shape.shape[1]] = trimmed_shape

            states.append(state*(stone_id+1))
        
        shape = np.rot90(shape)

    return states

if __name__ == "__main__":

    for stone in range(len(tetris_shapes)):
        for i in range(4*COLS):
            state = get_stone_states(stone)[i]
            state_ = np.where(state == 0, '.', state)  # Replace 0's with '.'
            state = np.where(state_ != '.', '0', state_)  # Replace 1's with 0
            print(state)
            print()