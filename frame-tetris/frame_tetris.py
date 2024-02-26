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
from state_gen import get_stone_states

BUFFER_SIZE = 3

all_stone_states = []
for stone in range(len(tetris_shapes)):
    all_stone_states.append(get_stone_states(stone))

class TetrisApp(object):
    def __init__(self, gui=True, cell_size=CELL_SIZE, cols=COLS, rows=ROWS, window_pos=(0, 0)):
        self.gui = gui

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
        #self.next_stoneID = 5 # TODO: remove this line
        self.next_stone = tetris_shapes[self.next_stoneID]

        self.gameover = False
        self.moves_wo_drop = 0
        self.max_moves_wo_drop = cols * 2

        self.states = {}
        self.init_game()

    def remove_row(self, board, row):
        rows, cols = board.shape
        # Create a new row of zeros
        new_row = np.zeros((1, cols), dtype=int)
        # Concatenate the new row with all rows except the one to remove
        new_board = np.vstack((new_row, np.delete(board, obj=row, axis=0)))
        return new_board
    
    def new_board(self):
        board = np.zeros((self.rows, self.cols), dtype=int)

        # Add buffer to bottom of board to avoid index errors
        buffer = np.ones((BUFFER_SIZE, board.shape[1]), dtype=int)
        board = np.vstack((board, buffer))
        return board

    def new_stone(self):
        self.stoneID = self.next_stoneID
        self.stone = self.next_stone[:]

        self.next_stoneID = rand(len(tetris_shapes))
        #self.next_stoneID = 5 # TODO: remove this line
        self.next_stone = tetris_shapes[self.next_stoneID]

        self.stone_state = int(self.cols / 2 - len(self.stone[0]) / 2)
        self.stone_offset = 0

        # Get board state slice for stone
        self.stone_slice = all_stone_states[self.stoneID][self.stone_state]

        # Drop stone until slice touches top layer of blocks        
        non_zero_rows = np.all(self.board == 0, axis=1)
        top_row = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])

        self.stone_offset = top_row - self.stone_slice.shape[0]# if top_row - len(self.stone) > 0 else 0 # TODO: Make sure this is correct
        self.score += self.stone_offset

        if not self.is_valid_state(self.stone_slice, self.stone_offset):
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
                    try:
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
                    except IndexError: # TODO: Fix know bug that causes this
                        with open('./step-tetris/error-log/log.txt', 'a') as file:
                            prnt = "\n".join(map(str, matrix))
                            file.write("New error: \n" + prnt + '\n\n')

    def add_cl_lines(self, n):
        #linescores = [0, 40, 100, 300, 1200]
        linescores = [0, 200, 500, 1500, 6000] # TODO: increase reward for lines temporarily
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level * 6:
            self.level += 1

    def quit(self):
        self.center_msg("Exiting...")
        if self.gui: pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover:
            self.score += 1 if manual else 0
            self.stone_offset += 1

            if not self.is_valid_state(self.stone_slice, self.stone_offset):
                self.stone_offset -= 1
                self.board[self.stone_offset:self.stone_offset+self.stone_slice.shape[0], :] += self.stone_slice
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-BUFFER_SIZE]):
                        if 0 not in row:
                            self.board = self.remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)

                self.new_stone()
                return True
        return False

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def update_board(self):
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
            self.draw_matrix(self.stone_slice.tolist(), (0, self.stone_offset))
            self.draw_matrix(self.board.tolist(), (0, 0))
            self.draw_matrix(self.next_stone, (self.cols + 1, 2))

        pygame.display.update()

    # Determines if this state is possible 
    # Potentially faster replacement for check_collision
    def is_valid_state(self, stone_slice, offset):
    
        board = self.board.copy()
        stone = stone_slice.copy()

        board[board != 0] = 1
        stone[stone != 0] = 1

        board[offset:offset+stone.shape[0]] += stone

        if np.any(board > 1):
            return False

        return True       
    
    # Determines if this state is reachable from current state
    def is_reachable_state(self, stone_slice, offset): # TODO: impliment this
        return True

    # Rotations in states are represented as rotations from current stone position
    def get_possible_states(self):

        stone_states = all_stone_states[self.stoneID]

        for i in range(self.cols * 4):

            state = stone_states[i]

            if np.all(state == 0): # Occurs when state index is invalid for stone
                continue

            # Determine if this position is real or if just confirming final pos
            y = self.stone_offset
            real = self.is_valid_state(state, y)

            # Determine if this possition is reachable from current state
            reachable = False
            if real:
                reachable = self.is_reachable_state(state, y)

            self.states[i] = (reachable, state, y)
    
        keys = np.array(list(self.states.keys()))
        values = np.array([v[0] for v in self.states.values()])

        # Create an array with zeros
        mask = np.zeros(keys.max() + 1, dtype=int)

        # Set positions to 1 based on condition
        mask[keys[values]] = 1

        # Return mask to apply to output vector of ai
        return mask

    def ai_command(self, state_id):

        state = self.states[state_id]

        # state is (possibility, stone, (x, y)) where possiblity is a bool, and rotation is stored in stone 2D list
        self.stone_slice = state[1]
        self.stone_offset = state[2]
        self.drop(True)

        if self.gui:
            self.update_board()

        sterile_board, _ = self.get_state()
        return sterile_board, self.next_stoneID, self.score, self.gameover
    
    def get_state(self):

        # Convert to 1s and 2s
        sterile_board = np.where(self.board != 0, 1, 0)
        sterile_stone = np.where(self.stone_slice != 0, 2, 0)

        # Add converted stone to board
        sterile_board[self.stone_offset:self.stone_offset+sterile_stone.shape[0], :] += sterile_stone

        return sterile_board.tolist(), self.next_stoneID

def print_board(board):
    sys.stdout.write('\033[F' * len(board))  # Move cursor up to overwrite previous lines
    for row in board:
        print(row)

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    App = TetrisApp()

    App.update_board()

    #print("\n"*ROWS)

    while not App.gameover:
        
        mask = App.get_possible_states()
        keys = np.where(mask == 1)[0]
        print(keys)
        
        val = input("Enter selection: ")
        if val == " ":
            App.quit()
            break

        if len(val) <= 2 and val.isdigit():
            val = int(val)
        else:
            print("Invalid input. Please enter a 2-digit number.")

        board, piece, score, gameover = App.ai_command(val)

        print(np.array(board))

        #print_board(board)

    profiler.disable()
    profiler.dump_stats("./step-tetris/profile_data.prof")
    stats_file = "./step-tetris/profile_data.prof"
    directory = './step-tetris/'
    utils.filter_methods(stats_file, directory)
