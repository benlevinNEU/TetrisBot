from multiprocessing import Process
import random
import numpy as np
import time
import sys, os
from step_tetris import TetrisApp

import cProfile
import pstats

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils

def game_controller(id, gui, window_pos, cell_size, rows, cols):
    app = TetrisApp(gui=gui, cell_size=cell_size, cols=cols, rows=rows, window_pos=window_pos)
    while True:
        command = random.randint(0, 3)
        board, piece, score, gameover = app.ai_command(command)
        if gameover:
            print(f"Process {id} - Game Over! Final score: {score}")
            break

def manage_games(num_games, gui):
    active_processes = {}
    completed_ids = set()

    cell_size = 10
    rows = 22
    cols = 10
    width = cell_size * (cols + 6)
    height = cell_size * rows + 80

    max_games_on_screen = int(2560 / width) * int(1440 / height)

    for i in range(min(num_games, max_games_on_screen)):
        start_game(i, active_processes, gui, width, height, cell_size, rows, cols)

    games_to_start = num_games - max_games_on_screen

    # Monitor active processes and restart completed ones as needed
    while active_processes:

        for id, process in list(active_processes.items()):
            if not process.is_alive():
                print(f"Restarting process {id}")
                completed_ids.add(id)
                process.join()  # Ensure the process has cleaned up resources
                del active_processes[id]  # Remove process from active processes
                # Optional: Restart the game with the same ID
                start_game(i, active_processes, width, height, cell_size, rows, cols)
                games_to_start -= 1

        if games_to_start <= 0:
            
            for id, process in active_processes.items():
                process.join()

            break

        time.sleep(0.1)

def start_game(id, active_processes, gui, width, height, cell_size, rows, cols):

    window_pos = ((width * id) % 2560, height * int(id / int(2560 / width)))

    # Create and start a new game process
    p = Process(target=game_controller, args=(id, gui, window_pos, cell_size, rows, cols))
    p.start()
    active_processes[id] = p  # Add to active processes

if __name__ == "__main__":
    default_num_games = 50  # Default number of games

    profile = True
    gui = False

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    
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
    
    manage_games(num_games, gui)

    print('End of main process')

    if profile:
        profiler.disable()
        profiler.dump_stats("./step-tetris/profile_data.prof")
        stats_file = "./step-tetris/profile_data.prof"
        directory = './step-tetris/'
        utils.filter_methods(stats_file, directory).print_stats()
