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
    from pynput import keyboard
    from pynput.keyboard import Key

BUFFER_SIZE = 4

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

class TetrisApp(object):
    def __init__(self, gui=True, cell_size=CELL_SIZE, cols=COLS, rows=ROWS, sleep=0.01, window_pos=(0, 0)):
        self.gui = gui
        self.sleep = sleep

        if gui:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '{},{}'.format(window_pos[0], window_pos[1])  # Set window position to '0
            pygame.init()
            pygame.key.set_repeat(250, 25)
        self.gui = gui
        self.cell_size = cell_size
        self.cols = cols
        self.rows = rows

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
        self.moves_wo_drop = 0
        self.max_moves_wo_drop = cols * 2

        self.states = {}
        self.init_game()
    
    def new_board(self):
        board = np.zeros((self.rows, self.cols), dtype=int)

        # Add buffer to bottom of board to avoid index errors
        buffer = np.ones((BUFFER_SIZE, board.shape[1]), dtype=int)
        board = np.vstack((buffer, board, buffer))

        # Add buffer to sides of board to avoid index errors
        buffer = np.ones((board.shape[0], BUFFER_SIZE), dtype=int)
        board = np.hstack((buffer, board, buffer))
        return board

    def new_stone(self):
        self.stoneID = self.next_stoneID
        self.stone = self.next_stone[:]

        self.next_stoneID = rand(len(tetris_shapes))
        self.next_stone = np.array(tetris_shapes[self.next_stoneID], dtype=int)

        self.stone_x = int((self.board.shape[1] - 2 * BUFFER_SIZE) / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0

        # Drop stone until slice touches top layer of blocks        
        non_zero_rows = np.all(self.board[BUFFER_SIZE:-BUFFER_SIZE, BUFFER_SIZE:-BUFFER_SIZE] == 0, axis=1)
        top_row = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])

        self.stone_y = max(top_row - self.stone.shape[0], 0)
        self.score += self.stone_y

        # If stone is line, allows stone to start at very top
        if not self.is_valid_state(self.stone, self.stone_x, self.stone_y)[0]:
            if self.stoneID == 5 and self.is_valid_state(self.stone, self.stone_x, self.stone_y-1)[0]:
                self.stone_y -= 1
                return

            self.gameover = True

    def init_game(self):
        self.board = self.new_board()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.new_stone()

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

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level * 6:
            self.level += 1

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
                self.draw_matrix(self.trimBoard(self.board).tolist(), (0, 0))
            else:
                self.draw_matrix(board.tolist(), (0, 0))
            self.draw_matrix(self.next_stone, (self.cols + 1, 2))

        pygame.display.update()

        #print(self.trimBoard(self.board).tolist() if board is None else self.trimBoard(board).tolist())

    # Preforms BFS from current state to find all possible finishing states for board
    def getFinalStates(self):

        visited_states = set()  # To track visited states
        final_boards = []       # To store final board states
        queue = [(self.stone, self.stone_x, self.stone_y, [])]  # Initial queue with starting state and board
        current_board = self.board.copy()  # Store the current board state

        if current_board.shape != (self.rows + 2*BUFFER_SIZE, self.cols + 2*BUFFER_SIZE):
            pass

        while queue:
            current_stone, current_x, current_y, steps = queue.pop(0)  # Dequeue an element
            for i, action in enumerate(actions):
                new_stone, new_x, new_y = action(current_stone, current_x, current_y)
                valid, touching_bottom = self.is_valid_state(new_stone, new_x, new_y) # Assuming this function exists and checks if the move is valid
                if valid:
                    state_key = (tuple(map(tuple, new_stone)), new_x, new_y)  # Convert to a hashable state
                    if state_key not in visited_states:
                        visited_states.add(state_key)
                        if touching_bottom:
                            new_board = current_board.copy()
                            new_board[BUFFER_SIZE+new_y:BUFFER_SIZE+new_y+new_stone.shape[0], 
                                      BUFFER_SIZE+new_x:BUFFER_SIZE+new_x+new_stone.shape[1]] += new_stone

                            final_boards.append((self.trimBoard(new_board), steps + [i]))  # Store the potential final board state and steps to get there
                            
                        queue.append((new_stone, new_x, new_y, steps + [i]))  # Enqueue new state

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

        board[BUFFER_SIZE+y:BUFFER_SIZE+y+stone.shape[0], 
              BUFFER_SIZE+x:BUFFER_SIZE+x+stone.shape[1]] += stone

        if np.any(board == 2):
            return False, False

        return True, np.any(board == 4)

    def trimBoard(self, board):
        return board[BUFFER_SIZE:-BUFFER_SIZE, BUFFER_SIZE:-BUFFER_SIZE]

    def ai_command(self, choice):

        (board, actions_) = choice

        if self.gui:
            for action in actions_:
                self.stone, self.stone_x, self.stone_y = actions[action](self.stone, self.stone_x, self.stone_y)
                self.update_board()
                pygame.time.wait(int(self.sleep*1000))

            self.board[BUFFER_SIZE+self.stone_y:BUFFER_SIZE+self.stone_y+self.stone.shape[0],
                       BUFFER_SIZE+self.stone_x:BUFFER_SIZE+self.stone_x+self.stone.shape[1]] += self.stone

        else:
            self.board[BUFFER_SIZE:-BUFFER_SIZE, BUFFER_SIZE:-BUFFER_SIZE] = board

        cleared_rows = np.sum(np.all(board != 0, axis=1))

        # Remove rows without zeros and add zeros at the top
        self.board[BUFFER_SIZE:-BUFFER_SIZE, BUFFER_SIZE:-BUFFER_SIZE] = np.vstack((np.zeros((cleared_rows, board.shape[1]), dtype=int), 
                                                                                    board[~np.all(board != 0, axis=1)]))

        self.add_cl_lines(cleared_rows)

        self.new_stone()

        if self.gui:
            self.update_board()

        return self.getFinalStates(), self.gameover, self.score

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
