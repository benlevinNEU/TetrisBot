from multiprocessing import Process
import random
import time
from step_tetris import TetrisApp  # Adjust import as needed

def game_controller(id, window_pos):
    app = TetrisApp(gui=True, cell_size=18, window_pos=window_pos)
    while True:
        command = random.randint(0, 3)
        board, piece, score, gameover = app.ai_command(command)
        if gameover:
            print(f"Process {id} - Game Over! Final score: {score}")
            break
        time.sleep(0.003)  # Adjust as needed for game speed

def manage_games(num_games):
    active_processes = {}
    completed_ids = set()

    for i in range(num_games):
        start_game(i, active_processes)

    # Monitor active processes and restart completed ones as needed
    while active_processes:
        for id, process in list(active_processes.items()):
            if not process.is_alive():
                print(f"Restarting process {id}")
                completed_ids.add(id)
                process.join()  # Ensure the process has cleaned up resources
                del active_processes[id]  # Remove process from active processes
                # Optional: Restart the game with the same ID
                start_game(id, active_processes)
        time.sleep(1)  # Check every second

def start_game(id, active_processes):
    cell_size = 18
    rows = 22
    cols = 10
    width = cell_size * (cols + 6)
    height = cell_size * rows
    window_pos = ((width * id) % (9 * width), (height + 60) * int(id / 9))

    # Create and start a new game process
    p = Process(target=game_controller, args=(id, window_pos,))
    p.start()
    active_processes[id] = p  # Add to active processes

if __name__ == "__main__":
    num_games = 27  # Number of game instances you want to run
    manage_games(num_games)
