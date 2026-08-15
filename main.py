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
tqdm.write(
    f"Training REINFORCE on a {global_defs.MAZE_HEIGHT}x{global_defs.MAZE_WIDTH} maze "
    f"({global_defs.SAMPLING_QUANTITY} episodes/batch)"
)
tqdm.write("-" * 78)
tqdm.write(f"{'epoch':>5}  {'loss':>10}  {'success':>8}  {'avg return':>11}  {'best return':>11}  {'temp':>7}")
tqdm.write("-" * 78)

for i in tqdm(range(global_defs.EPOCHS), desc="epochs"):
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
    tqdm.write(
        f"{i:5d}  "
        f"{float(loss.numpy()):10.4f}  "
        f"{metrics['success_rate']:8.2%}  "
        f"{metrics['avg_return']:11.3f}  "
        f"{metrics['best_return']:11.3f}  "
        f"{reinforce.temp:7.3f}"
    )
    reinforce.update_randomness()

    if metrics["success_rate"] >= 1.0:
        tqdm.write("-" * 78)
        tqdm.write(f"Solved: sampled training success reached 100% at epoch {i}.")
        break

final_eval = evaluate_sampled(reinforce, 100)
tqdm.write("-" * 78)
tqdm.write("Final evaluation")
tqdm.write(f"  success rate: {final_eval['success_rate']:.2%}")
tqdm.write(f"  avg return:   {final_eval['avg_return']:.3f}")
tqdm.write(f"  best return:  {final_eval['best_return']:.3f}")
#print(reinforce.trajectories)
