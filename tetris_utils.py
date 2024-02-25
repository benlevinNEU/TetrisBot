from random import randrange as rand
import numpy as np
import sys, os

def print_board(board):
    sys.stdout.write('\033[F' * len(board))  # Move cursor up to overwrite previous lines
    for row in board:
        print(row)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = open(os.devnull, 'w')  # Redirect standard output to /dev/null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()  # Close the file handle
        sys.stdout = self._original_stdout  # Restore the original standard output

with SuppressOutput():
    import pygame

# The configuration
CELL_SIZE = 18
COLS = 10
ROWS = 22
maxfps = 144

colors = [
    (0, 0, 0),
    (255, 85, 85),
    (100, 200, 115),
    (120, 108, 245),
    (255, 140, 50),
    (50, 120, 52),
    (146, 202, 73),
    (150, 161, 218),
    (35, 35, 35),  # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
    [[0, 1, 0], 
     [1, 1, 1], 
     [0, 0, 0]],

    [[0, 2, 2], 
     [2, 2, 0], 
     [0, 0, 0]],

    [[3, 3, 0], 
     [0, 3, 3], 
     [0, 0, 0]],

    [[4, 0, 0], 
     [4, 4, 4], 
     [0, 0, 0]],

    [[0, 0, 5], 
     [5, 5, 5], 
     [0, 0, 0]],

    [[0, 0, 0, 0], 
     [6, 6, 6, 6], 
     [0, 0, 0, 0], 
     [0, 0, 0, 0]],

    [[7, 7], 
     [7, 7]],
]

def rotate_clockwise(shape):
    shape = np.rot90(shape).tolist()
    return shape

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            if cell != 0:
                # Check if the current position is within the bounds of the board
                if 0 <= cy + off_y < len(board) and 0 <= cx + off_x < len(board[0]):
                    if board[cy + off_y][cx + off_x]:
                        return True
                else:
                    # If out of bounds, it's a collision
                    return True
    return False

# trims 0's around shapes for colision detection
def trim(matrix):
    matrix = np.array(matrix)
    
    # Find the indices where the trimming should stop
    row_nonzero = np.where(matrix.any(axis=1))[0]
    col_nonzero = np.where(matrix.any(axis=0))[0]
    
    if row_nonzero.size > 0:
        matrix = matrix[row_nonzero[0]:row_nonzero[-1]+1, :]
    if col_nonzero.size > 0:
        matrix = matrix[:, col_nonzero[0]:col_nonzero[-1]+1]
        
    return matrix.tolist()

# joins shape to board
def join_matrixes(mat1, mat2, mat2_off):

    left_shift, right_shift, up_shift, down_shift = getShifts(mat2)
    mat2 = trim(mat2)
    
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            try:
                mat1[cy + off_y + up_shift][cx + off_x + left_shift] += val #TODO: Check this
            except IndexError:
                pass
    return mat1

def getShifts(shape):
    shape_array = np.array(shape)
    
    # Identifying non-zero rows and columns
    non_zero_cols = np.any(shape_array != 0, axis=0)
    non_zero_rows = np.any(shape_array != 0, axis=1)
    
    # Calculating shifts
    left_shift = np.argmax(non_zero_cols)
    right_shift = len(non_zero_cols) - np.argmax(non_zero_cols[::-1])
    up_shift = np.argmax(non_zero_rows)
    down_shift = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])
    
    return left_shift, right_shift, up_shift, down_shift