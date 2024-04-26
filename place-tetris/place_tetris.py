#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tetris_utils import *
import utils
import cProfile
#from state_gen import get_stone_states

import platform
import time

pltfm = None
if platform.system() == 'Linux' and 'microsoft-standard-WSL2' in platform.release():
    pltfm = 'WSL'
    import curses
    #import keyboard
else:
    pltfm = 'Mac'
    from pynput import keyboard # type: ignore
    from pynput.keyboard import Key # type: ignore
import threading

BUF_SZ = 4

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models/")

actions = [
    lambda stone, x, y: (stone, x + 1, y),
    lambda stone, x, y: (stone, x - 1, y),
    lambda stone, x, y: (np.rot90(stone), x, y),
    lambda stone, x, y: (np.rot90(stone, k=3), x, y),
    lambda stone, x, y: (stone, x, y + 1)
]

def trimBoard(board):
    return board[BUF_SZ:-BUF_SZ, BUF_SZ:-BUF_SZ]

class TetrisApp(object):
    def __init__(self, gui=True, cell_size=CELL_SIZE, cols=COLS, rows=ROWS, sleep=0.01, window_pos=(0, 0), state=None, snap=None):

        self.gui = gui
        self.sleep = sleep

        self.gui = gui
        self.cell_size = cell_size
        self.cols = cols
        self.rows = rows

        # Phantom game that's played when evaluating moves for next stone
        if state is not None:
            self.board, self.stone, self.stone_x, self.stone_y, self.score, self.next_stoneID, self.steps = state
            self.phantom = True

            self.window_pos = window_pos

            self.next_stone = np.array(tetris_shapes[self.next_stoneID], dtype=int)
            self.lines = 0
            self.level = 1
            self.gameover = False
            return
        
        self.phantom = False

        if gui:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '{},{}'.format(window_pos[0], window_pos[1])  # Set window position to '0
            pygame.init()
            pygame.key.set_repeat(250, 25)


        self.width = cell_size * (cols + 6)
        self.height = cell_size * rows
        self.rlim = cell_size * cols
        self.bground_grid = [
            [8 if x % 2 == y % 2 else 0 for x in range(cols)] for y in range(rows)
        ]

        if gui:
            self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.event.set_blocked(pygame.MOUSEMOTION)
        # mouse movement
        # events, so we
        # block them.
        self.next_stoneID = rand(len(tetris_shapes))
        self.next_stone = np.array(tetris_shapes[self.next_stoneID], dtype=int)

        self.gameover = False

        self.states = {}

        if snap is not None:
            self.board, self.score, self.level, self.lines = snap
            self.new_stone(True)
            return

        self.init_game()
        
    def new_board(self):
        board = np.zeros((self.rows, self.cols), dtype=int)

        # Add buffer to bottom of board to avoid index errors
        buffer = np.ones((BUF_SZ, board.shape[1]), dtype=int)
        board = np.vstack((buffer, board, buffer))

        # Add buffer to sides of board to avoid index errors
        buffer = np.ones((board.shape[0], BUF_SZ), dtype=int)
        board = np.hstack((buffer, board, buffer))
        return board

    def new_stone(self, demo=False):
        self.stoneID = self.next_stoneID
        self.stone = self.next_stone[:]

        self.next_stoneID = rand(len(tetris_shapes))
        self.next_stone = np.array(tetris_shapes[self.next_stoneID], dtype=int)

        self.stone_x = int((self.board.shape[1] - 2 * BUF_SZ) / 2 - len(self.stone[0]) / 2)
        stone_y = 0

        # Drop stone until slice touches top layer of blocks        
        non_zero_rows = np.all(self.board[BUF_SZ:-BUF_SZ, BUF_SZ:-BUF_SZ] == 0, axis=1)
        top_row = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])

        stone_y = max(top_row - self.stone.shape[0], 0)

        # If stone is line, allows stone to start at very top
        if not self.is_valid_state(self.stone, self.stone_x, stone_y)[0]:
            #if self.stoneID == 5 and self.is_valid_state(self.stone, self.stone_x, stone_y-1)[0]:
            #    stone_y -= 1
            #
            #else:
            self.gameover = True

        if not demo:
            self.stone_y = stone_y
            self.score += stone_y
        else:
            self.stone_y = 0
            return stone_y

    def init_game(self):
        self.board = self.new_board()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.new_stone(True)

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)),
                (x, y),
            )
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(
                line, False, (255, 255, 255), (0, 0, 0)
            )

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(
                msg_image,
                (
                    self.width // 2 - msgim_center_x,
                    self.height // 2 - msgim_center_y + i * 22,
                ),
            )

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):

                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x + x) * self.cell_size,
                            (off_y + y) * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                        0,
                    )

    def clear_rows(self):
        tB = trimBoard(self.board)
        cleared_rows = np.sum(np.all(tB != 0, axis=1))

        # Remove rows without zeros and add zeros at the top
        self.board[BUF_SZ:-BUF_SZ, BUF_SZ:-BUF_SZ] = np.vstack((np.zeros((cleared_rows, tB.shape[1]), dtype=int), 
                                                                tB[~np.all(tB != 0, axis=1)]))

        points = self.add_cl_lines(cleared_rows)
        return points, self.board.copy()

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        points = linescores[n] * self.level
        self.score += points
        if self.lines >= self.level * 6:
            self.level += 1

        return points

    def quit(self):
        self.center_msg("Exiting...")
        if self.gui: pygame.display.update()
        sys.exit()

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def update_board(self, board=None):
        self.screen.fill((0, 0, 0))
        if self.gameover:
            self.center_msg(
                """Game Over!\nYour score: %d Press space to continue""" % self.score
            )
        else:
            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (self.rlim + 1, 0),
                (self.rlim + 1, self.height - 1),
            )
            self.disp_msg("Next:", (self.rlim + self.cell_size, 2))
            self.disp_msg(
                "Score: %d\n\nLevel: %d\nLines: %d"
                % (self.score, self.level, self.lines),
                (self.rlim + self.cell_size, self.cell_size * 5),
            )
            self.draw_matrix(self.bground_grid, (0, 0))
            
            if board is None:
                self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
                self.draw_matrix(trimBoard(self.board).tolist(), (0, 0))
            else:
                self.draw_matrix(board.tolist(), (0, 0))
            self.draw_matrix(self.next_stone, (self.cols + 1, 2))

        pygame.display.update()

        #print(trimBoard(self.board).tolist() if board is None else trimBoard(board).tolist())

    # Preforms BFS from current state to find all possible finishing states for board for current stone and next stone
    # Uses cost as heuristic to cut off proportion of worst real board states
    # Ensures that state_key is unique when evaluating phantom states prevents dupicates from being evaluated
    def getFinalStates(self, vs_key=None, cp=None, stone_y=None):

        if stone_y is None:
            stone_y = self.stone_y

        if vs_key is None:
            visited_states, prev_state_key = set(), None
        else:
            visited_states, prev_state_key = vs_key # To track visited states and last state key

        final_boards = []       # To store final board states (where second stone touches bottom)
        mid_boards = []         # To store board states where first stone touches bottom
        queue = [(self.stone, self.stone_x, stone_y, [])]  # Initial queue with starting state and board
        current_board = self.board.copy()  # Store the current board state

        while queue:
            current_stone, current_x, current_y, steps = queue.pop(0)  # Dequeue an element
            for i, action in enumerate(actions):
                new_stone, new_x, new_y = action(current_stone, current_x, current_y)
                valid, touching_bottom = self.is_valid_state(new_stone, new_x, new_y) # Assuming this function exists and checks if the move is valid
                if valid:
                    if self.phantom:
                        state_key_partial1 = prev_state_key
                        state_key_partial2 = (tuple(map(tuple, new_stone)), new_x, new_y)  # Convert to a hashable state
                        state_key = (state_key_partial1, state_key_partial2)  # Convert to a hashable state
                    else:
                        state_key = ((tuple(map(tuple, new_stone)), new_x, new_y), None)

                    # Check if state is in either order
                    if state_key not in visited_states and (state_key[1], state_key[0]) not in visited_states:
                        visited_states.add(state_key)
                        if touching_bottom:
                            new_board = current_board.copy()
                            new_board[BUF_SZ+new_y:BUF_SZ+new_y+new_stone.shape[0], 
                                      BUF_SZ+new_x:BUF_SZ+new_x+new_stone.shape[1]] += new_stone

                            # Initialize phantom game to simulate next stone
                            # Steps to get to this point are stored in phantom game so steps accumulated phantom can be ignored
                            phantom = TetrisApp(gui=False, 
                                                cell_size=self.cell_size, cols=self.cols, rows=self.rows, sleep=self.sleep, 
                                                state=(new_board, new_stone, new_x, new_y, 0, self.next_stoneID, steps + [i]))
                            
                            points, new_board = phantom.clear_rows()
                            points_scored = points + (self.score if self.phantom else 0)

                            if self.phantom:
                                # Phantom board, Real board, actions, points
                                # Steps to get to real game are stored. Steps accumulated in phantom game are ignored
                                option = (trimBoard(new_board), trimBoard(self.board), self.steps, points_scored)
                                final_boards.append(option)
                            
                            else:
                                cost = None
                                if cp is not None:
                                    model = cp[0]
                                    option = (trimBoard(new_board), None, None, points_scored)
                                    cost, _, _, _ = model.cost(option, *cp[1:])

                                phantom.new_stone()
                                if not phantom.gameover:
                                    mid_boards.append((phantom, state_key[0], cost))
                            
                        queue.append((new_stone, new_x, new_y, steps + [i]))  # Enqueue new state

        # Check if any board has a cost that is None
        if mid_boards != []:
            if any(board[2] is None for board in mid_boards):
                # If any board has a cost of None, process all boards
                boards_to_process = mid_boards
            else:
                # Proccess only the top (1 - PRUNE_RATIO) of boards
                sorted_mid_boards = sorted(mid_boards, key=lambda x: x[2])  # x[2] is the cost
                prune_ratio = cp[1]['prune_ratio']
                boards_to_process = sorted_mid_boards[:int(len(sorted_mid_boards) * (1 - prune_ratio))]

            # Process the selected boards
            for board in boards_to_process:
                phantom, state_key, _ = board  # _ is used to ignore the cost in the unpacking as it's not needed here
                final_boards.extend(phantom.getFinalStates(vs_key=(visited_states, state_key)))

        if final_boards == [] and not self.phantom:
            pass
        return final_boards

    # Determines if this state is possible and returns if stone touching bottom
    # VERY PROUD OF HOW BEAUTIFUL THIS METHOD IS
    def is_valid_state(self, stone, x, y):
    
        board = self.board.copy()
        stone = stone.copy()

        board[board != 0] = 1
        stone[stone != 0] = 1

        stone = np.vstack((stone, np.zeros((1, stone.shape[1]), dtype=int)))

        non_zero_rows = np.all(stone == 0, axis=1)
        insert_at = stone.shape[0] - np.argmin(non_zero_rows[::-1])

        stone[insert_at] = stone[insert_at-1]*3

        board[BUF_SZ+y:BUF_SZ+y+stone.shape[0], 
              BUF_SZ+x:BUF_SZ+x+stone.shape[1]] += stone

        if np.any(board == 2):
            return False, False

        return True, np.any(board == 4)

    def ai_command(self, choice, cp=None):

        (f_board, r_board, actions_, _) = choice

        if self.gui:
            for action in actions_:
                self.stone, self.stone_x, self.stone_y = actions[action](self.stone, self.stone_x, self.stone_y)
                if action == 4:
                    self.score += 1
                self.update_board()
                pygame.time.wait(int(self.sleep*1000))

            self.board[BUF_SZ+self.stone_y:BUF_SZ+self.stone_y+self.stone.shape[0],
                       BUF_SZ+self.stone_x:BUF_SZ+self.stone_x+self.stone.shape[1]] += self.stone

        else:
            self.board[BUF_SZ:-BUF_SZ, BUF_SZ:-BUF_SZ] = r_board
            self.score += actions_.count(4)

        self.clear_rows()
        y = self.new_stone(cp[1]['demo'])

        if self.gui:
            self.update_board()

        if self.gui and cp[1]['demo']:
            def run_in_thread(container):
                finalStates = self.getFinalStates(cp=cp, stone_y=y)
                container.append(finalStates)

            container = []
            thread = threading.Thread(target=run_in_thread, args=(container,))
            thread.start()

            while self.stone_y != y:
                self.stone_y += 1
                self.score += 1
                self.update_board()
                pygame.time.wait(int(self.sleep*1000))

            thread.join()
            finalStates = container[0]
        
        else:
            finalStates = self.getFinalStates(cp=cp)

        snapshot = (self.board.copy(), self.score, self.level, self.lines)

        return finalStates, self.gameover, self.score, snapshot

    def quit_game(self):
        pygame.quit()

def print_board(board):
    sys.stdout.write('\033[F' * len(board))  # Move cursor up to overwrite previous lines
    for row in board:
        print(row)

if __name__ == "__main__":

    selection = None
    profile = False

    App = TetrisApp(sleep=0.1, gui=True)

    App.update_board()

    options = App.getFinalStates()
    print(len(options))
    choice = int(len(options) / 2)

    while not App.gameover:
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        options, _, _ = App.ai_command(options[choice])
                        choice = int(len(options) / 2)
                    elif event.key == pygame.K_LEFT:
                        choice = max(0, choice - 1)
                        App.update_board(options[choice][0])
                    elif event.key == pygame.K_RIGHT:
                        choice = min(choice + 1, len(options) - 1)
                        App.update_board(options[choice][0])
                    elif event.key == pygame.K_ESCAPE:
                        sys.exit()

            pygame.key.set_repeat()
