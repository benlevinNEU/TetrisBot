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

NUM_EVALS = 9

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

class Evals():

    def __init__(self, GP):
        self.GP = GP

        # Get Theoretical Maxes
        self.MXH = GP['rows']
        self.MX_DTE = GP['cols']/2
        self.MX_BMPS = ((GP['cols']-1)*GP['rows'])

        self.HS = np.zeros(GP['cols'])
        self.HS[[0,-1]] = self.MXH
        self.MX_POLY_COEF = np.polyfit(np.arange(GP['cols']), self.HS, 2)[0]

        self.MX_HOLES = GP["rows"] * GP["cols"] / 2
        self.MX_OVERHANGS = GP["rows"] * GP["cols"] / 2

        self.MAX_POINTS = 1200 + GP['rows'] - 4

        # Fear of Death Warning
        self.FoD_W = 4

    def heights(self, arr):
        # Get the index of the first 1 in each column from the top
        idxs = np.argmax(arr, axis=0)
        heights = self.MXH - idxs

        # Check for columns that are all zeros and set their heights to 0
        heights[arr.max(axis=0) == 0] = 0
        
        return heights

    def getHP(self, board):

        hs = self.heights(board)

        bmps = np.sum(abs(np.diff(hs)))
        
        min_height = np.min(hs)
        max_height = np.max(hs)
        avg_height = np.mean(hs)

        mx_h4e = np.min([np.min(hs), self.GP['cols'] - np.max(hs)])
        mn_h4e = np.min([np.max(hs), self.GP['cols'] - np.min(hs)])

        coef = np.polyfit(np.arange(self.GP['cols']), hs, 2)[0]

        n_bmps = bmps / self.MX_BMPS
        n_max_height = max_height / self.MXH
        n_avg_height = avg_height / self.MXH
        n_min_height = min_height / self.MXH
        n_mx_h4e = mx_h4e / self.MX_DTE
        n_mn_h4e = mn_h4e / self.MX_DTE
        n_coef = coef / self.MX_POLY_COEF

        fod = min(int(self.MXH - max_height), self.FoD_W) / self.FoD_W

        return n_bmps, n_max_height, n_avg_height, n_min_height, n_coef, fod

    def getHoles(self, board):

        trimboard = board.copy()
        h = max(0, np.argmax(np.any(trimboard == 1, axis=1)) - 1)
        l = max(0, np.argmax(np.any(trimboard == 1, axis=0)) - 1)
        r = min(self.GP["cols"], self.GP["cols"] - np.argmax(np.any(trimboard[:, ::-1] == 1, axis=0)) + 1)

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

        return island_count / self.MX_HOLES, trimboard

    def getOverhangs(self, board):
        overhangs = 0
        pattern = np.array([1, 0])

        for col in range(board.shape[1]):
            column = board[:, col]
            windows = np.lib.stride_tricks.sliding_window_view(column, window_shape=2)
            for window in windows:
                if np.array_equal(window, pattern):
                    overhangs += 1

        return overhangs / self.MX_OVERHANGS

    def getPointsForMove(self, state):

        board, actions = state

        actions = np.array(actions)

        drops = np.sum(actions == 3)
        cl_rows = np.sum(np.all(board != 0, axis=1))

        # TODO: Make extensible
        linescores = [0, 40, 100, 300, 1200]
        cl_pnts = linescores[cl_rows]

        points = cl_pnts + drops

        max_points = 1200 + self.GP['rows'] - 4

        return points / max_points

    def getEvals(self, state):

        # Phantom board, Real board, actions, points
        board, _, _, points = state

        board = board.copy()
        board[board > 0] = 1

        #points = self.getPointsForMove(state)
        n_points = points / self.MAX_POINTS

        cleared_rows = np.sum(np.all(board != 0, axis=1))
        board = np.vstack((np.zeros((cleared_rows, board.shape[1]), dtype=int), board[~np.all(board != 0, axis=1)]))

        bmps, max_height, avg_height, min_height, n_coef, fod = self.getHP(board)

        holes, board = self.getHoles(board)  # Outputs board with holes plugged with 2's
        overhangs = self.getOverhangs(board)

        vals = np.array([bmps, max_height, avg_height, min_height, holes, overhangs, n_points, n_coef, fod])

        return vals

def getEvalLabels():
    return ["bmps   ", "mx_h   ", "av_h   ", "mn_h   ", "holes  ", "ovhangs", "points ", "poly_c ", "FoD_W  "]

# Tests
if __name__ == "__main__":
    evals = Evals(GP)
    print(evals.getEvals((board, [])))

    # Example usage
    array = np.array([
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ])
    assert (evals.heights(array) == np.array([2, 2, 3, 2])).all()
    vals = evals.getHP(array)
    assert (vals[0] == 2)
    assert (vals[1] == 2)

    array = np.array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ])
    assert (evals.heights(array) == np.array([2, 2, 5, 2])).all()
    vals = evals.getHP(array)
    assert (vals[0] == 6)
    assert (vals[1] == 2)

    array = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ])
    assert (evals.heights(array) == np.array([5, 2, 4, 2])).all()
    vals = evals.getHP(array)
    assert (vals[0] == 7)


    array = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ])
    assert (evals.heights(array) == np.array([5, 2, 4, 0])).all()
    vals = evals.getHP(array)
    assert (vals[0] == 9)
    assert (vals[1] == 0)

    array = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ])
    assert (evals.heights(array) == np.array([5, 2, 2, 6])).all()
    vals = evals.getHP(array)
    assert (vals[0] == 7)

    print("All tests passed!")