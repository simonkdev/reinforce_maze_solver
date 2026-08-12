import tensorflow as tf
import numpy as np
from maze_utils import mazes

generator = mazes.KruskalMaze(n_x=20, n_y=20)
maze = generator.maze

print(maze)

print("Hello World")
