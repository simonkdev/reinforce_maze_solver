import numpy as np

MAZE_WIDTH = 20  # generator adds two columns for borders
MAZE_HEIGHT = 20  # generator adds two rows for borders
# right, left, up, down
ACTIONS = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
EPISODE_LIMIT = 30
