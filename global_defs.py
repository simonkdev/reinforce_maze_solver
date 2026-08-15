import numpy as np

MAZE_WIDTH = 20  # generator adds two columns for borders
MAZE_HEIGHT = 20  # generator adds two rows for borders
# right, left, up, down
ACTIONS = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
EPISODE_LIMIT = 100
SAMPLING_QUANTITY = 64
DISCOUNT_VALUE = 0.99
EPOCHS = 120
SUCCESS_REWARD = 10.0
PENALTY = -0.01
PROGRESS_REWARD = 1.0
LEARNING_RATE = 0.03
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
TEMP_DECAY = 0.02
EVAL_TEMP = 0.05
