import numpy as np

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
                       

def getHeightParams(board):
    bmps = 0
    max_height = 0

    column = board[:, 0]
    loc = np.where(column[::-1]==1)
    h1 = np.max(loc)+1 if len(loc[0]) > 0 else 0
    height_sum = h1

    # Will be used for trimming board for following tests
    mt_lcols = board.shape[1] # Subtract 1 to make sure holes doesn't get false positive
    mt_rcols = board.shape[1]

    for col in range(1, board.shape[1]):
        if h1 > 0:
            # Subtract 1 to make sure holes doesn't get false positive
            mt_lcols = min(col-2, mt_lcols)
            mt_rcols = max(board.shape[1] - col - 1, 0)

        if h1 > max_height:
            max_height = h1

        column = board[:, col]
        loc = np.where(column[::-1]==1)
        h2 = np.max(loc)+1 if len(loc[0]) > 0 else 0
        bmps += abs(h1 - h2)
        h1 = h2

        height_sum += h1

    if h2 > max_height:
        max_height = h2

    mt_lcols = max(mt_lcols, 0)

    start_index = max(board.shape[0]-max_height-1, 0)

    if board[start_index:board.shape[0], mt_lcols:board.shape[1]-mt_rcols].shape[0] < 2:
        pass
    
    # Impliment board trimming later
    #trimmed_board = board[start_index:board.shape[0], mt_lcols:board.shape[1]-mt_rcols]

    return bmps, max_height, height_sum/board.shape[1], board #trimmed_board

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

    visited = np.ones_like(board, dtype=bool)*-1
    visited[board == 1] = 0

    island_count = 0

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0 and visited[i, j] == -1:
                if i == 0: # Top row should be empty and not filled 
                    dfs(board, j, i, visited, 0)
                else: 
                    dfs(board, j, i, visited, 0, fill=2)
                    island_count += 1

    board += visited

    # TODO: Make sure not nonetypes

    return island_count, board

def getOverhangs(board):
    overhangs = 0
    pattern = np.array([1, 0])

    for col in range(board.shape[1]):
        column = board[:, col]
        windows = np.lib.stride_tricks.sliding_window_view(column, window_shape=2)
        for window in windows:
            if np.array_equal(window, pattern):
                overhangs += 1

    return overhangs

def getPointsForMove(state):

    board, actions = state

    board[board == 8] = 0
    board[board == 5] = 0

    actions = np.array(actions)

    drops = np.count_nonzero(actions == 3)
    cl_rows = np.count_nonzero(np.all(board != 0, axis=1))

    # TODO: Make extensible
    linescores = [0, 40, 100, 300, 1200]
    cl_pnts = linescores[cl_rows]

    return drops + cl_pnts

def getEvals(state):

    board, _ = state

    board = board.copy()
    board[board > 0] = 1

    points = getPointsForMove(state)

    bmps, max_height, avg_height, board = getHeightParams(board) # Outputs trimmed board

    holes, board = getHoles(board)  # Outputs board with holes plugged with 2's
    overhangs = getOverhangs(board)

    return bmps, max_height, avg_height, holes, overhangs, points

if __name__ == "__main__":
    print(getEvals((board, [])))