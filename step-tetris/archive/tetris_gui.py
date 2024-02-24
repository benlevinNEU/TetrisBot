import pygame
import numpy as np
import random
import json
from tetris import Tetris

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

class TetrisGUI(Tetris):
    def __init__(self, height=20, width=10):
        super().__init__(height, width)
        pygame.init()
        self.cell_size = 30  # Size of one square cell in pixels
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()  # To control game speed

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(self.screen, 
                                     (255, 255, 255), 
                                     pygame.Rect((off_x + x) * self.cell_size, 
                                                 (off_y + y) * self.cell_size, 
                                                 self.cell_size, self.cell_size), 0)

    def draw_game(self):
        self.screen.fill((0, 0, 0))
        # Draw the board
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] == 1:
                    pygame.draw.rect(self.screen, 
                                     (255, 255, 255), 
                                     pygame.Rect(x * self.cell_size, 
                                                 y * self.cell_size, 
                                                 self.cell_size, self.cell_size), 0)
        # Draw the current piece
        if self.current_piece is not None:
            piece_matrix = self.current_piece
            piece_offset = self.piece_position[1], self.piece_position[0]
            self.draw_matrix(piece_matrix, piece_offset)
        
        pygame.display.flip()

    def run(self):
        key_actions = {
            'ESCAPE':   lambda: self._exit(),
            'LEFT':     lambda: self.play_step(1),
            'RIGHT':    lambda: self.play_step(2),
            'DOWN':     lambda: self.play_step(0),
            'UP':       lambda: self.play_step(3),
            'SPACE':    lambda: self.play_step(4),
        }

        self.new_piece()
        while not self.game_over:
            self.clock.tick(30)  # Limit to 30 FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._exit()
                elif event.type == pygame.KEYDOWN:
                    for key in key_actions:
                        if event.key == getattr(pygame, f'K_{key}'):
                            key_actions[key]()
            
            self.draw_game()

        print("Game Over! Score:", self.score)
        pygame.time.wait(2000)  # Wait 2 seconds before closing
        self._exit()

    def _exit(self):
        pygame.quit()
        quit()

# Running the game with GUI
if __name__ == "__main__":
    game = TetrisGUI()
    game.run()
