from multiprocessing import Process
import random
import numpy as np
import time
import sys
from step_tetris import TetrisApp  # Adjust import as needed

def game_controller(id, window_pos, cell_size, rows, cols):
    app = TetrisApp(gui=True, cell_size=cell_size, cols=cols, rows=rows, window_pos=window_pos)
    while True:
        command = random.randint(0, 3)
        board, piece, score, gameover = app.ai_command(command)
        if gameover:
            print(f"Process {id} - Game Over! Final score: {score}")
            break

def manage_games(num_games):
    active_processes = {}
    completed_ids = set()

    cell_size = 10
    rows = 22
    cols = 10
    width = cell_size * (cols + 6)
    height = cell_size * rows + 80

    max_games_on_screen = int(2560 / width) * int(1440 / height)

    for i in range(min(num_games, max_games_on_screen)):
        start_game(i, active_processes, width, height, cell_size, rows, cols)

    games_to_start = num_games - max_games_on_screen

    # Monitor active processes and restart completed ones as needed
    while active_processes:

        if games_to_start <= 0:
            break

        for id, process in list(active_processes.items()):
            if not process.is_alive():
                print(f"Restarting process {id}")
                completed_ids.add(id)
                process.join()  # Ensure the process has cleaned up resources
                del active_processes[id]  # Remove process from active processes
                # Optional: Restart the game with the same ID
                start_game(i, active_processes, width, height, cell_size, rows, cols)
                games_to_start -= 1
        time.sleep(1)  # Check every second

def start_game(id, active_processes, width, height, cell_size, rows, cols):

    window_pos = ((width * id) % 2560, height * int(id / int(2560 / width)))

    # Create and start a new game process
    p = Process(target=game_controller, args=(id, window_pos, cell_size, rows, cols))
    p.start()
    active_processes[id] = p  # Add to active processes

if __name__ == "__main__":
    default_num_games = 50  # Default number of games
    
    # Check if the correct argument is provided
    if len(sys.argv) >= 2 and sys.argv[1] == "-n":
        try:
            num_games = int(sys.argv[2])
        except (IndexError, ValueError):
            print("Invalid number of games. Using default.")
            num_games = default_num_games
    else:
        print(f"No input provided. Using default number of games: {default_num_games}")
        num_games = default_num_games
    
    manage_games(num_games)