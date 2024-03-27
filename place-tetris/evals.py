import numpy as np
from local_params import GP

board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
                 [0, 0, 0, 0, 0, 0, 0, 5, 1, 1], 
                 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 
                 [0, 0, 0, 1, 5, 0, 0, 1, 1, 8], 
                 [0, 0, 0, 1, 0, 0, 1, 1, 8, 8], 
                 [1, 1, 0, 1, 0, 0, 1, 1, 8, 8], 
                 [8, 1, 1, 1, 1, 1, 1, 8, 8, 1], 
                 [8, 1, 8, 1, 8, 1, 1, 8, 1, 1], 
                 [1, 1, 1, 1, 8, 1, 1, 1, 1, 8]])

board[board == 8] = 0
board[board == 5] = 0

board_empty_edges = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 5, 1, 0], 
                             [0, 0, 0, 1, 1, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 1, 5, 0, 0, 1, 1, 0], 
                             [0, 0, 0, 1, 0, 0, 1, 1, 5, 0], 
                             [0, 0, 0, 1, 0, 0, 1, 1, 0, 0], 
                             [0, 0, 1, 1, 1, 1, 1, 5, 0, 0], 
                             [0, 0, 5, 1, 8, 1, 1, 0, 1, 0], 
                             [0, 0, 1, 1, 8, 1, 1, 1, 1, 0]])

board_empty_edges[board_empty_edges == 8] = 0
board_empty_edges[board_empty_edges == 5] = 0

test_holes = np.array([[0, 0, 0, 0],
                       [0, 0, 1, 1],
                       [1, 1, 0, 0],
                       [0, 1, 1, 0]])
                       

def getHeightParams():
    bmps = 0
    max_height = 0
    min_height = np.inf
    mx_h4e = int(GP["cols"]/2)
    mn_h4e = 0

    column = board[:, 0]                            # Vertical slice of board
    loc = np.where(column[::-1]==1)
    h1 = np.max(loc)+1 if len(loc[0]) > 0 else 0
    height_sum = h1

    for col in range(1, GP["cols"]):

        if h1 > max_height:
            max_height = h1
            mx_h4e = min(col - 0, GP["cols"] - col) # Max height distance from edge

        if h1 < min_height:
            min_height = h1
            mn_h4e = min(col - 0, GP["cols"] - col) # Min height distance from edge

        column = board[:, col]
        loc = np.where(column[::-1]==1)
        h2 = np.max(loc)+1 if len(loc[0]) > 0 else 0
        bmps += abs(h1 - h2)
        h1 = h2

        height_sum += h1

    if h2 > max_height:
        max_height = h2

    # Get theoretical max bmps
    max_bmps = ((GP['cols']-1)*GP['rows'])

    n_bmps = bmps/max_bmps
    n_max_height = max_height/GP['rows']
    n_avg_height = height_sum/GP['cols']/GP['rows']
    n_min_height = min_height/GP['rows']
    n_mx_h4e = mx_h4e/(GP['cols']/2)
    n_mn_h4e = mn_h4e/(GP['cols']/2)

    # Return normalized values
    return n_bmps, n_max_height, n_avg_height, n_min_height, n_mx_h4e, n_mn_h4e

def dfs(matrix, x, y, visited, value, fill=0):

    if y < 0 or y >= matrix.shape[0] or x < 0 or x >= matrix.shape[1]:
        return # Out of bounds
    if visited[y, x] != -1 or matrix[y, x] != value:
        return # Already visited or not part of an island
    
    # Mark the current cell as visited
    visited[y, x] = fill

    # Explore neighbors (up, down, left, right)
    dfs(matrix, x + 1, y, visited, value, fill)
    dfs(matrix, x - 1, y, visited, value, fill)
    dfs(matrix, x, y + 1, visited, value, fill)
    dfs(matrix, x, y - 1, visited, value, fill)

    return matrix

def getHoles(board):

    trimboard = board.copy()
    h = max(0, np.argmax(np.any(trimboard == 1, axis=1)) - 1)
    l = max(0, np.argmax(np.any(trimboard == 1, axis=0)) - 1)
    r = min(GP["cols"], GP["cols"] - np.argmax(np.any(trimboard[:, ::-1] == 1, axis=0)) + 1)

    trimboard = trimboard[h:, l:r]

    if trimboard.shape[0] == 1:
        pass

    visited = np.ones_like(trimboard, dtype=bool)*-1
    visited[trimboard == 1] = 0

    island_count = 0

    for i in range(trimboard.shape[0]):
        for j in range(trimboard.shape[1]):
            if trimboard[i, j] == 0 and visited[i, j] == -1:
                if i == 0: # Top row should be empty and not filled 
                    dfs(trimboard, j, i, visited, 0)
                else: 
                    dfs(trimboard, j, i, visited, 0, fill=2)
                    island_count += 1

    trimboard += visited

    max_holes = GP["rows"] * GP["cols"] / 2

    return island_count/max_holes, trimboard

def getOverhangs(board):
    overhangs = 0
    pattern = np.array([1, 0])

    for col in range(board.shape[1]):
        column = board[:, col]
        windows = np.lib.stride_tricks.sliding_window_view(column, window_shape=2)
        for window in windows:
            if np.array_equal(window, pattern):
                overhangs += 1

    max_overhangs = GP["rows"] * GP["cols"] / 2

    return overhangs / max_overhangs

def getPointsForMove(state):

    board, actions = state

    actions = np.array(actions)

    drops = np.sum(actions == 3)
    cl_rows = np.sum(np.all(board != 0, axis=1))

    # TODO: Make extensible
    linescores = [0, 40, 100, 300, 1200]
    cl_pnts = linescores[cl_rows]

    points = cl_pnts + drops

    max_points = 1200 + GP['rows'] - 4

    return points / max_points

NUM_EVALS = 9
def getEvals(state):

    board, _ = state

    board = board.copy()
    board[board > 0] = 1

    points = getPointsForMove(state)

    cleared_rows = np.sum(np.all(board != 0, axis=1))
    board = np.vstack((np.zeros((cleared_rows, board.shape[1]), dtype=int), board[~np.all(board != 0, axis=1)]))

    bmps, max_height, avg_height, min_height, mx_h4e, mn_h4e = getHeightParams()

    holes, board = getHoles(board)  # Outputs board with holes plugged with 2's
    overhangs = getOverhangs(board)

    vals = np.array([bmps, max_height, avg_height, min_height, holes, overhangs, points, mx_h4e, mn_h4e])

    return vals

if __name__ == "__main__":
    print(getEvals((board, [])))