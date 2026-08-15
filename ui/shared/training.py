from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
import time
from typing import Callable

import numpy as np
import tensorflow as tf
from maze_utils import mazes

import global_defs
from policy import Policy
from reinforce import Reinforce


ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = global_defs.EPOCHS
    sampling_quantity: int = global_defs.SAMPLING_QUANTITY
    episode_limit: int = global_defs.EPISODE_LIMIT
    eval_episodes: int = 100
    seed: int | None = None


def default_maze(inner_height: int = 20, inner_width: int = 20) -> list[list[int]]:
    maze = np.zeros((inner_height + 2, inner_width + 2), dtype=int)
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1
    maze[inner_height // 2 + 1, 0] = 2
    maze[inner_height // 2 + 1, inner_width + 1] = 3
    return maze.tolist()


def random_maze(inner_height: int = 20, inner_width: int = 20, seed: int | None = None) -> list[list[int]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    generator = mazes.KruskalMaze(n_x=inner_height, n_y=inner_width)
    return generator.maze.astype(int).tolist()


def validate_maze(maze: list[list[int]]) -> np.ndarray:
    matrix = np.array(maze, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("Maze must be a two-dimensional matrix.")
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError("Maze must include padding and at least one interior cell.")

    values = set(np.unique(matrix).tolist())
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError("Maze cells must be 0=open, 1=wall, 2=start, or 3=end.")

    starts = np.argwhere(matrix == 2)
    ends = np.argwhere(matrix == 3)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Maze must contain exactly one start and one end.")

    start = tuple(int(item) for item in starts[0])
    end = tuple(int(item) for item in ends[0])
    if not _is_padding_cell(start, matrix.shape):
        raise ValueError("Start must be placed in the padding border.")
    if not _is_padding_cell(end, matrix.shape):
        raise ValueError("End must be placed in the padding border.")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if _is_padding_cell((row, col), matrix.shape) and matrix[row, col] == 0:
                raise ValueError("Padding border must be walls except for start and end.")

    return matrix


def train_from_matrix(
    maze: list[list[int]],
    config: TrainConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    config = config or TrainConfig()
    matrix = validate_maze(maze)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    with _temporary_global_config(matrix, config):
        start = np.array([np.array(np.where(matrix == 2)).flatten()])
        end = np.array([np.array(np.where(matrix == 3)).flatten()])
        policy = Policy()
        agent = Reinforce(policy, start, end, matrix)
        start_point = (int(start[0][0]), int(start[0][1]))
        if start_point not in agent.distance_map:
            raise ValueError("Start cannot reach the end through open cells.")

        history = []
        started_at = time.perf_counter()
        for epoch in range(config.epochs):
            agent.sample_run()
            metrics = agent.get_progress_metrics()

            with tf.GradientTape() as tape:
                loss = agent.get_policy_loss()

            gradients = tape.gradient(loss, policy.model.trainable_variables)
            policy.optimizer.apply_gradients(zip(gradients, policy.model.trainable_variables))
            agent.update_randomness()

            greedy_progress = _greedy_rollout(agent)

            row = {
                "epoch": epoch,
                "loss": float(loss.numpy()),
                "success_rate": float(metrics["success_rate"]),
                "avg_return": float(metrics["avg_return"]),
                "best_return": float(metrics["best_return"]),
                "temperature": float(agent.temp),
                "seconds": time.perf_counter() - started_at,
                "path_success": bool(greedy_progress["success"]),
                "path_steps": int(greedy_progress["steps"]),
                "shortest_path": int(greedy_progress["shortest_path"]),
            }
            history.append(row)
            if on_progress:
                on_progress({**row, "total_epochs": config.epochs})

            if metrics["success_rate"] >= 1.0:
                break

        final_eval = _evaluate_sampled(agent, config.eval_episodes)
        greedy = _greedy_rollout(agent)
        elapsed = time.perf_counter() - started_at

    return {
        "maze": matrix.tolist(),
        "history": history,
        "final_eval": final_eval,
        "greedy": greedy,
        "epochs_run": len(history),
        "seconds": elapsed,
        "shortest_path": int(greedy["shortest_path"]),
    }


def _is_padding_cell(point: tuple[int, int], shape: tuple[int, int]) -> bool:
    x, y = point
    return x == 0 or y == 0 or x == shape[0] - 1 or y == shape[1] - 1


@contextmanager
def _temporary_global_config(matrix: np.ndarray, config: TrainConfig):
    saved = {
        "MAZE_HEIGHT": global_defs.MAZE_HEIGHT,
        "MAZE_WIDTH": global_defs.MAZE_WIDTH,
        "EPISODE_LIMIT": global_defs.EPISODE_LIMIT,
        "SAMPLING_QUANTITY": global_defs.SAMPLING_QUANTITY,
        "EPOCHS": global_defs.EPOCHS,
    }
    global_defs.MAZE_HEIGHT = matrix.shape[0] - 2
    global_defs.MAZE_WIDTH = matrix.shape[1] - 2
    global_defs.EPISODE_LIMIT = config.episode_limit
    global_defs.SAMPLING_QUANTITY = config.sampling_quantity
    global_defs.EPOCHS = config.epochs
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(global_defs, key, value)


def _evaluate_sampled(agent: Reinforce, episodes: int) -> dict:
    old_quantity = global_defs.SAMPLING_QUANTITY
    old_temp = agent.temp
    global_defs.SAMPLING_QUANTITY = episodes
    agent.temp = global_defs.EVAL_TEMP
    agent.sample_run()
    metrics = agent.get_progress_metrics()
    global_defs.SAMPLING_QUANTITY = old_quantity
    agent.temp = old_temp
    return {key: float(value) for key, value in metrics.items()}


def _greedy_rollout(agent: Reinforce) -> dict:
    old_temp = agent.temp
    agent.temp = global_defs.EVAL_TEMP
    agent.prepare_policy_mask()

    state = agent.maze_start.copy()
    path = [[int(state[0][0]), int(state[0][1])]]
    start_point = (int(state[0][0]), int(state[0][1]))
    shortest_path = agent.distance_map[start_point]

    for steps in range(global_defs.EPISODE_LIMIT):
        if np.array_equal(state, agent.maze_end):
            agent.temp = old_temp
            return {
                "success": True,
                "steps": steps,
                "path": path,
                "shortest_path": shortest_path,
            }

        x = int(state[0][0])
        y = int(state[0][1])
        action_index = int(np.argmax(agent.policy_matrix[x, y]))
        state = state + global_defs.ACTIONS[action_index]
        path.append([int(state[0][0]), int(state[0][1])])

    success = bool(np.array_equal(state, agent.maze_end))
    agent.temp = old_temp
    return {
        "success": success,
        "steps": global_defs.EPISODE_LIMIT,
        "path": path,
        "shortest_path": shortest_path,
    }
