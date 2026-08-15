from dataclasses import dataclass
from collections import deque

import tensorflow as tf
import numpy as np
import global_defs
from tqdm import tqdm

@dataclass
class timestep:
    initial_state: np.ndarray
    action: np.ndarray
    action_index: int
    action_prob: float
    new_state: np.ndarray
    new_reward: float
    expected_reward: float

@dataclass
class compact_timestep:
    expected_reward: float
    action: np.ndarray
    action_prob: float

@dataclass
class precursor_loss_object:
    total_cum_reward: float
    timestep_amount: int
    action_prob: float

class Reinforce:
    def __init__(self, policy, start, end, maze):
         self.hello = "Hello World"
         self.policy = policy
         self.terminate_trajectory = False
         self.steps = 0
         self.timesteps = []
         self.trajectories = []
         self.need_clear_trajectories = False
         self.trajectory_count = 0
         self.maze_start = start
         self.maze_end = end
         self.state = start
         self.maze = maze
         self.temp = global_defs.INITIAL_TEMP
         self.distance_map = self.calculate_distance_map()

    def calculate_distance_map(self):
        end = (int(self.maze_end[0][0]), int(self.maze_end[0][1]))
        distances = {end: 0}
        queue = deque([end])

        while queue:
            x, y = queue.popleft()

            for action in global_defs.ACTIONS:
                nx = x + int(action[0])
                ny = y + int(action[1])
                point = (nx, ny)

                if point in distances:
                    continue

                if (
                    0 <= nx < self.maze.shape[0]
                    and 0 <= ny < self.maze.shape[1]
                    and self.maze[nx][ny] != 1
                ):
                    distances[point] = distances[(x, y)] + 1
                    queue.append(point)

        return distances

    def get_policy_loss(self):
        self.calculate_cumulative_rewards()
        return self.calculate_average_policy_loss()

    def get_progress_metrics(self):
        returns = []
        successes = 0

        for trajectory in self.trajectories:
            episode_return = sum(step.new_reward for step in trajectory)
            returns.append(episode_return)

            if np.array_equal(trajectory[-1].new_state, self.maze_end):
                successes += 1

        return {
            "best_return": max(returns),
            "avg_return": sum(returns) / len(returns),
            "success_rate": successes / len(self.trajectories),
        }

    def sample_run(self):
        self.prepare_policy_mask()
        if(self.need_clear_trajectories):
            self.trajectories = []
            self.need_clear_trajectories = False

        for i in tqdm(range(global_defs.SAMPLING_QUANTITY)):
            self.trajectory()

        self.trajectory_count = 0
        self.need_clear_trajectories = True



    def parse_maze_matrix(self):
        self.maze_mask = np.tile([0.0,0.0,0.0,0.0], (self.maze.shape[0], self.maze.shape[1], 1))
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i][j] == 1:
                    self.maze_mask[i][j] = [1, 1, 1, 1]
                    if((i-1) >= 0):
                        self.maze_mask[i-1][j][3] = 1
                    if((i+1) <= self.maze.shape[0]-1):
                        self.maze_mask[i+1][j][2] = 1
                    if((j-1) >= 0):
                        self.maze_mask[i][j-1][0] = 1
                    if((j+1) <= self.maze.shape[1]-1):
                        self.maze_mask[i][j+1][1] = 1
                if self.maze[i][j] == 2:
                    if self.maze_start[0][1] == 0:
                        self.maze_mask[i][j][1] = 1
                    if self.maze_start[0][1] == (self.maze.shape[1] - 1):
                        self.maze_mask[i][j][0] = 1
                    if self.maze_start[0][0] == 0:
                        self.maze_mask[i][j][2] = 1
                    if self.maze_start[0][0] == (self.maze.shape[0] - 1):
                        self.maze_mask[i][j][3] = 1

    def mask_policy_matrix(self):
        self.policy_matrix = np.copy(self.raw_policy_matrix)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                mask = self.maze_mask[i][j]
                idx = 0
                for int in mask:
                    if (int == 1):
                        self.policy_matrix[i][j][idx] = -1e11
                    idx += 1

    def populate_policy_matrix(self):
        coords = np.indices(self.maze.shape).reshape(2, -1).T.astype(np.float32)
        predictions = self.policy.model(coords).numpy()
        self.raw_policy_matrix = predictions.reshape(
            self.maze.shape[0],
            self.maze.shape[1],
            len(global_defs.ACTIONS),
        )

    def prepare_policy_mask(self):
        self.parse_maze_matrix()
        self.populate_policy_matrix()
        self.mask_policy_matrix()
        self.policy_matrix = tf.nn.softmax(self.policy_matrix / self.temp, axis=-1).numpy()

    def decide(self, probabilities):
        probs_np = np.asarray(probabilities)
        index = np.random.choice(len(global_defs.ACTIONS), p=probs_np)
        return global_defs.ACTIONS[index], index, float(probs_np[index])

    def get_action_for_state(self, local_state):
        state_x = local_state[0][0]
        state_y = local_state[0][1]
        probabilities = self.policy_matrix[state_x, state_y]
        action, action_index, action_prob = self.decide(probabilities)
        return action, action_index, action_prob

    def get_reward_for_new_state(self, new_state):
        current = (int(self.state[0][0]), int(self.state[0][1]))
        new = (int(new_state[0][0]), int(new_state[0][1]))

        progress = self.distance_map[current] - self.distance_map[new]
        reward = global_defs.PENALTY + (global_defs.PROGRESS_REWARD * progress)

        if np.array_equal(new_state, self.maze_end):
            reward += global_defs.SUCCESS_REWARD

        return reward

    def timestep(self):
        current_state = self.state
        action, action_index, action_prob = self.get_action_for_state(current_state)

        new_state = current_state + action
        reward = self.get_reward_for_new_state(new_state)

        self.state = new_state

        t = timestep(
            current_state,
            action,
            action_index,
            action_prob,
            new_state,
            reward,
            0.0
        )
        self.timesteps.append(t)

        if np.array_equal(new_state, self.maze_end):
            self.terminate_trajectory = True

    def trajectory(self):
        self.state = self.maze_start.copy()
        self.steps = 0

        while not self.terminate_trajectory:
            self.timestep()
            self.steps += 1

            if self.steps >= global_defs.EPISODE_LIMIT:
                self.terminate_trajectory = True

        self.trajectories.append(self.timesteps)
        self.timesteps = []
        self.terminate_trajectory = False

    def calculate_cumulative_rewards(self):
        for trajectory in tqdm(self.trajectories):
            reward_buffer = 0
            for i in range(len(trajectory)):
                expected = (reward_buffer * global_defs.DISCOUNT_VALUE) + trajectory[len(trajectory) - 1 - i].new_reward
                trajectory[len(trajectory) - 1 - i].expected_reward = expected
                reward_buffer = expected

    def assemble_cumulative_state_rewards(self):
        self.cum_state_matrix = np.empty((self.maze.shape[0], self.maze.shape[1]), dtype=object)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                self.cum_state_matrix[i, j] = []

        for trajectory in tqdm(self.trajectories):
            for timestep in trajectory:
                state_x = timestep.initial_state[0][0]
                state_y = timestep.initial_state[0][1]
                compact = compact_timestep(timestep.expected_reward, timestep.action, timestep.action_prob)
                self.cum_state_matrix[state_x, state_y].append(compact)

    def assemble_precursor_loss(self):
        self.precursor_matrix = np.empty((self.maze.shape[0], self.maze.shape[1]), dtype=object)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                self.precursor_matrix[i, j] = [0, 0, 0, 0]


        for i in range(self.cum_state_matrix.shape[0]):
            for j in range(self.cum_state_matrix.shape[1]):

                state_list = self.cum_state_matrix[i, j]

                action1_sum = 0.0
                action1_amount = 0
                action1_probability = 0.0
                action1_prob_grabbed = False

                action2_sum = 0.0
                action2_amount = 0
                action2_probability = 0.0
                action2_prob_grabbed = False

                action3_sum = 0.0
                action3_amount = 0
                action3_probability = 0.0
                action3_prob_grabbed = False

                action4_sum = 0.0
                action4_amount = 0
                action4_probability = 0.0
                action4_prob_grabbed = False

                for ct in state_list:
                    act = ct.action
                    if(np.array_equal(act, global_defs.ACTIONS[0])):
                        action1_sum = action1_sum + ct.expected_reward
                        action1_amount = action1_amount + 1
                        if not action1_prob_grabbed:
                            action1_probability = ct.action_prob
                            action1_prob_grabbed = True
                    elif(np.array_equal(act, global_defs.ACTIONS[1])):
                        action2_sum = action2_sum + ct.expected_reward
                        action2_amount = action2_amount + 1
                        if not action2_prob_grabbed:
                            action2_probability = ct.action_prob
                            action2_prob_grabbed = True
                    elif(np.array_equal(act, global_defs.ACTIONS[2])):
                        action3_sum = action3_sum + ct.expected_reward
                        action3_amount = action3_amount + 1
                        if not action3_prob_grabbed:
                            action3_probability = ct.action_prob
                            action3_prob_grabbed = True
                    elif(np.array_equal(act, global_defs.ACTIONS[3])):
                        action4_sum = action4_sum + ct.expected_reward
                        action4_amount = action4_amount + 1
                        if not action4_prob_grabbed:
                            action4_probability = ct.action_prob
                            action4_prob_grabbed = True

                self.precursor_matrix[i][j][0] = precursor_loss_object(action1_sum, action1_amount, action1_probability)
                self.precursor_matrix[i][j][1] = precursor_loss_object(action2_sum, action2_amount, action2_probability)
                self.precursor_matrix[i][j][2] = precursor_loss_object(action3_sum, action3_amount, action3_probability)
                self.precursor_matrix[i][j][3] = precursor_loss_object(action4_sum, action4_amount, action4_probability)

    def calculate_average_policy_loss(self):
        steps = [step for trajectory in self.trajectories for step in trajectory]
        states = np.array([step.initial_state[0] for step in steps], dtype=np.float32)
        action_indices = np.array([step.action_index for step in steps], dtype=np.int32)
        rewards = np.array([step.expected_reward for step in steps], dtype=np.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-11)
        state_indices = states.astype(np.int64)
        masks = self.maze_mask[state_indices[:, 0], state_indices[:, 1]]

        logits = self.policy.model(tf.convert_to_tensor(states, dtype=tf.float32))
        mask_tensor = tf.convert_to_tensor(masks, dtype=tf.bool)
        masked_logits = tf.where(
            mask_tensor,
            tf.fill(tf.shape(logits), -1e9),
            logits,
        )
        log_probs = tf.nn.log_softmax(masked_logits / self.temp)
        selected_log_probs = tf.gather(
            log_probs,
            tf.convert_to_tensor(action_indices, dtype=tf.int32),
            axis=1,
            batch_dims=1,
        )

        return tf.reduce_mean(-selected_log_probs * tf.convert_to_tensor(rewards, dtype=tf.float32))

    def update_randomness(self):
        self.temp = max(global_defs.MIN_TEMP, self.temp * (1 - global_defs.TEMP_DECAY))
