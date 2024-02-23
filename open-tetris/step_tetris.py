#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Very simple tetris implementation
#
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#     Return - Instant drop
#
# Have fun!

# NOTE: If you're looking for the old python2 version, see
#       <https://gist.github.com/silvasur/565419/45a3ded61b993d1dd195a8a8688e7dc196b08de8>

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from random import randrange as rand
import pygame, sys
import numpy as np
import sys

# The configuration
cell_size = 18
cols = 10
rows = 22
maxfps = 30

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
    [[0, 0, 0], 
     [1, 1, 1], 
     [0, 1, 0]],

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

    # TODO: Check edge oclussions

    return shape


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell != 0 and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board

def trim(matrix):

    matrix = np.array(matrix)

    while np.all(matrix[0] == 0):
        matrix = matrix[1:]
    while np.all(matrix[-1] == 0):
        matrix = matrix[:-1]
    while np.all(matrix[:, 0] == 0):
        matrix = matrix[:, 1:]
    while np.all(matrix[:, -1] == 0):
        matrix = matrix[:, :-1]
    return matrix.tolist()

def join_matrixes(mat1, mat2, mat2_off):

    mat2 = trim(mat2)
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            try:
                mat1[cy + off_y - 1][cx + off_x] += val
            except IndexError:
                pass
    return mat1


def new_board():
    board = [[0 for x in range(cols)] for y in range(rows)]
    board += [[1 for x in range(cols)]]
    return board


class TetrisApp(object):
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.width = cell_size * (cols + 6)
        self.height = cell_size * rows
        self.rlim = cell_size * cols
        self.bground_grid = [
            [8 if x % 2 == y % 2 else 0 for x in range(cols)] for y in range(rows)
        ]

        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need
        # mouse movement
        # events, so we
        # block them.
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]

        self.gameover = False
        self.init_game()

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(cols / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0

        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.board = new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        # pygame.time.set_timer(pygame.USEREVENT+1, 1000)

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

                if val > 8: # TODO: For debugging purposes
                    pass

                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x + x) * cell_size,
                            (off_y + y) * cell_size,
                            cell_size,
                            cell_size,
                        ),
                        0,
                    )

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level * 6:
            self.level += 1
            newdelay = 1000 - 50 * (self.level - 1)
            newdelay = 100 if newdelay < 100 else newdelay
            pygame.time.set_timer(pygame.USEREVENT + 1, newdelay)

    def move(self, delta_x):
        if not self.gameover:

            # Find columns where all elements are 0
            all_zeros = np.all(np.array(self.stone) == 0, axis=0)

            # Find the first column where not all elements are 0
            left_shift = np.argmax(all_zeros == False)
            right_shift = np.argmax(all_zeros[::-1] == False)

            new_x = self.stone_x + delta_x
            if new_x < -left_shift:
                new_x = -left_shift
            if new_x > cols - len(self.stone[0]) + right_shift:
                new_x = cols - len(self.stone[0]) + right_shift

            if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
                self.stone_x = new_x

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self):
        if not self.gameover:
            # self.score += 1
            self.stone_y += 1
            if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                    self.board, self.stone, (self.stone_x, self.stone_y)
                )
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False

    def insta_drop(self):
        if not self.gameover:
            while not self.drop(True):
                pass

    def rotate_stone(self):
        if not self.gameover:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
                self.stone = new_stone

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
            self.disp_msg("Next:", (self.rlim + cell_size, 2))
            self.disp_msg(
                "Score: %d\n\nLevel: %d\nLines: %d"
                % (self.score, self.level, self.lines),
                (self.rlim + cell_size, cell_size * 5),
            )
            self.draw_matrix(self.bground_grid, (0, 0))
            self.draw_matrix(self.board, (0, 0))
            self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
            self.draw_matrix(self.next_stone, (cols + 1, 2))

        pygame.display.update()

    def ai_command(self, command):
        key_actions = {
            0: self.drop,
            1: lambda: self.move(-1),
            2: lambda: self.move(+1),
            3: self.rotate_stone,
        }

        self.gameover = False

        if command in key_actions:
            key_actions[command]()

        self.update_board()

        piece = np.array(self.stone).max()
        sterile_board = np.where(np.array(self.board) != 0, 1, 0).tolist()
        sterile_stone = np.where(np.array(self.stone) != 0, 2, 0).tolist()

        sterile_board = join_matrixes(sterile_board, sterile_stone, (self.stone_x, self.stone_y))

        return sterile_board, piece, self.score, self.gameover

def print_board(board):
    sys.stdout.write('\033[F' * len(board))  # Move cursor up to overwrite previous lines
    for row in board:
        print(row)

if __name__ == "__main__":
    App = TetrisApp()

    App.update_board()

    print("\n"*rows)

    while not App.gameover:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    board, piece, score, gameover = App.ai_command(3)
                elif event.key == pygame.K_DOWN:
                    board, piece, score, gameover = App.ai_command(0)
                elif event.key == pygame.K_LEFT:
                    board, piece, score, gameover = App.ai_command(1)
                elif event.key == pygame.K_RIGHT:
                    board, piece, score, gameover = App.ai_command(2)

                print_board(board)

