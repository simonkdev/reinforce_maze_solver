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

def evaluate_sampled(agent, episodes):
    old_quantity = global_defs.SAMPLING_QUANTITY
    old_temp = agent.temp
    global_defs.SAMPLING_QUANTITY = episodes
    agent.temp = global_defs.EVAL_TEMP
    agent.sample_run()
    metrics = agent.get_progress_metrics()
    global_defs.SAMPLING_QUANTITY = old_quantity
    agent.temp = old_temp
    return metrics

losses = []
for i in range(global_defs.EPOCHS):
    reinforce.sample_run()
    metrics = reinforce.get_progress_metrics()

    with tf.GradientTape() as tape:
        loss = reinforce.get_policy_loss()

    gradients = tape.gradient(
        loss,
        policy.model.trainable_variables
    )
    policy.optimizer.apply_gradients(
        zip(gradients, policy.model.trainable_variables)
    )
    losses.append(float(loss.numpy()))
    print(i, float(loss.numpy()), metrics)
    reinforce.update_randomness()
print(losses)
print("final_eval", evaluate_sampled(reinforce, 100))
#print(reinforce.trajectories)
