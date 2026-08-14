from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from maze_utils import mazes
from policy import Policy
import global_defs
from tqdm import tqdm
from reinforce import Reinforce

generator = mazes.KruskalMaze(n_x=global_defs.MAZE_HEIGHT, n_y=global_defs.MAZE_WIDTH)

maze = generator.maze
START = np.array([np.array(np.where(maze == 2)).flatten()])
END = np.array([np.array(np.where(maze == 3)).flatten()])
policy = Policy()
reinforce = Reinforce(policy, START, END, maze)

losses = []
for i in range(global_defs.EPOCHS):
    with tf.GradientTape() as tape:
        reinforce.sample_run()
        loss = reinforce.get_policy_loss()
    losses.append(loss)
    gradients = tape.gradient(
        loss,
        policy.model.trainable_variables
    )
    policy.optimizer.apply_gradients(
        zip(gradients, policy.model.trainable_variables)
    )
print(losses)
