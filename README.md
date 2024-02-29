## Welcome to StackOverflow: Tetris AI

This project is our my Capstone for CS4100: Artificial Intelligence at Northeastern University. The goal of this project is to create an AI that can play Tetris. The AI will be trained using reinforcement learning and will be able to play the game at a high level.

Import the environment with the following command:
```bash
conda env create -f ./res/tetris.yml
```

Activate the environment with the following command:
```bash
conda activate tetris
```

To run step_tetris run the following command:
```bash
python ./place-tetris/place_tetris.py
```

To run the tetris training run the following command:
```bash
python ./place-tetris/ai.py
```
Make sure to push new generations to git when you are done training. If you want to leave your computer running, you can push at any time because the networks are saved continuously.

To run a specific ai model, edit the file to define your weights and run the following command:
```bash
python ./place-tetris/test-model.py
```