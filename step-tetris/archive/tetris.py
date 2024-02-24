import numpy as np
import random
import json

# The configuration
cell_size = 18
cols =      10
rows =      22
maxfps =    30

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

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

class Tetris:

    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width), dtype=int)
        self.game_over = False
        self.score = 0
        self.logs = []  # To keep track of actions and scores
        # Define pieces using 4 rotations for simplicity; in a full implementation, you would calculate rotations
        self.pieces = self.initialize_pieces()
        self.current_piece = None
        self.piece_position = (0, 0)
        self.new_piece()

    def initialize_pieces(self):
        # This is a simplified representation; you might want to use a more complex structure for rotations
        pieces = {
            'I': [np.array([[1, 1, 1, 1]])],
            'O': [np.array([[1, 1], [1, 1]])],
            'T': [np.array([[0, 1, 0], [1, 1, 1]])],
            'S': [np.array([[0, 1, 1], [1, 1, 0]])],
            'Z': [np.array([[1, 1, 0], [0, 1, 1]])],
            'L': [np.array([[1, 0], [1, 0], [1, 1]])],
            'J': [np.array([[0, 1], [0, 1], [1, 1]])]
        }
        return pieces
    
    def new_piece(self):
        self.current_piece = random.choice(list(self.pieces.values()))[0]
        self.piece_position = (0, self.width // 2 - len(self.current_piece[0]) // 2)

    def check_collision(self, offset=(0, 0)):
        # Offset is a tuple (y, x) representing the movement or rotation
        piece = self.current_piece
        offset_y, offset_x = offset
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y, x] == 1:
                    new_y = y + self.piece_position[0] + offset_y
                    new_x = x + self.piece_position[1] + offset_x
                    if new_x < 0 or new_x >= self.width or new_y >= self.height:
                        return True
                    if self.board[new_y, new_x] == 1:
                        return True
        return False

    def rotate_piece(self):
        # Simplified rotation that doesn't account for wall kicks or SRS.
        original_piece = self.current_piece.copy()
        self.current_piece = np.rot90(self.current_piece)
        if self.check_collision():
            self.current_piece = original_piece  # Revert if collision occurs post-rotation

    def move_piece(self, direction):
        # Move the piece left (-1) or right (+1) if no collision
        if not self.check_collision((0, direction)):
            self.piece_position = (self.piece_position[0], self.piece_position[1] + direction)
        # Automatically lock the piece if it cannot move down anymore
        elif direction == 0 and self.check_collision((1, 0)):
            self.lock_piece_and_clear_lines()

    def drop_piece(self):
        # Drop the piece one row, lock it if it collides
        if not self.check_collision((1, 0)):
            self.piece_position = (self.piece_position[0] + 1, self.piece_position[1])
        else:
            self.lock_piece_and_clear_lines()

    # lock_piece_and_clear_lines, clear_lines, log_current_state methods...

    def play_step(self, action):
        """
        Perform a single step in the game based on the given action.
        - 0: Do nothing
        - 1: Move left
        - 2: Move right
        - 3: Rotate
        - 4: Drop
        """
        if action == 1:
            self.move_piece(-1)
        elif action == 2:
            self.move_piece(1)
        elif action == 3:
            self.rotate_piece()
        elif action == 4:
            self.drop_piece()
        else:
            self.move_piece(0)  # Move down or lock

        self.check_game_over()

    def check_game_over(self):
        # Game over if the new piece has nowhere to go
        if self.check_collision((0, 0)) and self.piece_position[0] <= 1:
            self.game_over = True
            print("Game Over! Score:", self.score)
            self.save_logs()

    # Example manual play loop
    def manual_play(self):
        import time
        self.new_piece()
        while not self.game_over:
            self.print_board()
            action = input("Action (L/R/Rotate/Drop/Nothing): ").strip().lower()
            if action == 'l':
                self.play_step(1)
            elif action == 'r':
                self.play_step(2)
            elif action.startswith('rot'):
                self.play_step(3)
            elif action == 'drop':
                self.play_step(4)
            else:
                self.play_step(0)
            time.sleep(0.5)  # Slow down for readability

    def print_board(self):
        # Simple console print to visualize the board and piece
        temp_board = self.board.copy()
        for y in range(self.current_piece.shape[0]):
            for x in range(self.current_piece.shape[1]):
                if self.current_piece[y, x] == 1:
                    temp_board[y + self.piece_position[0], x + self.piece_position[1]] = 2
        print("\n".join(["".join(["X" if cell == 1 else ("1" if cell == 2 else "0") for cell in row]) for row in temp_board]))
        print("-" * self.width)

# Running the game manually for testing
if __name__ == "__main__":
    game = Tetris()
    game.manual_play()

# Placeholder for AI algorithm - you would implement decision-making logic here
class TetrisAI:
    def decide(self, board, piece):
        # Implement AI decision-making logic
        return random.choice(['left', 'right', 'rotate', 'drop'])

# Example of running the game in headless mode with AI
#game = Tetris()
#ai = TetrisAI()
#game.play(ai_algorithm=ai)
