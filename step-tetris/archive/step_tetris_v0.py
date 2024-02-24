import pygame, sys
from random import randrange as rand

# Configuration Variables
cols = 10
rows = 22
cell_size = 20

# Tetris Shapes
tetris_shapes = [
    [[1, 1, 1], [0, 1, 0]],
    [[0, 2, 2], [2, 2, 0]],
    [[3, 3, 0], [0, 3, 3]],
    [[4, 0, 0], [4, 4, 4]],
    [[0, 0, 5], [5, 5, 5]],
    [[6, 6, 6, 6]],
    [[7, 7], [7, 7]]
]

class TetrisApp:
    def __init__(self):
        pygame.init()
        self.width = cell_size * cols
        self.height = cell_size * rows
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.init_game()

    def new_board(self):
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def init_game(self):
        self.board = self.new_board()
        self.new_stone()
        self.gameover = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines = 0

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]

        self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0

        if self.check_collision(self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def check_collision(self, board, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and board[ cy + off_y ][ cx + off_x ]:
                        return True
                except IndexError:
                    return True
        return False

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = [[self.stone[y][x] for y in range(len(self.stone))] for x in range(len(self.stone[0]) - 1, -1, -1)]
            if not self.check_collision(new_stone, (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def join_matrixes(self, mat1, mat2, mat2_off):
        off_x, off_y = mat2_off
        for y, row in enumerate(mat2):
            for x, val in enumerate(row):
                mat1[y + off_y][x + off_x] += val
        return mat1

    def remove_row(self, row):
        del self.board[row]
        self.board.insert(0, [0 for _ in range(cols)])

    def drop(self):
        if not self.gameover and not self.paused:
            self.score += 1
            self.stone_y += 1
            if self.check_collision(self.stone, (self.stone_x, self.stone_y)):
                self.board = self.join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = self.remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False

    def update_game(self):
        self.drop()

    def draw(self):
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size), 0)
        pygame.display.update()

    def ai_command(self, command):
        """Process a command from the AI or control mechanism."""
        if command == "left":
            self.stone_x -= 1 if self.stone_x > 0 else 0
        elif command == "right":
            self.stone_x += 1 if self.stone_x < cols - len(self.stone[0]) else 0
        elif command == "rotate":
            self.rotate_stone()
        elif command == "drop":
            self.update_game()
        self.draw()

        return self.board, self.score, self.gameover

if __name__ == '__main__':
    App = TetrisApp()

    while not App.gameover:
        App.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    App.rotate_stone()
                elif event.key == pygame.K_DOWN:
                    App.drop()
                elif event.key == pygame.K_LEFT:
                    App.stone_x -= 1 if App.stone_x > 0 else 0
                elif event.key == pygame.K_RIGHT:
                    App.stone_x += 1 if App.stone_x < cols - len(App.stone[0]) else 0

        App.update_game()
        App.draw()
        pygame.time.wait(1000)