from dataclasses import dataclass

import tensorflow as tf
from tensorflow.dtypes import float32
import numpy as np
from maze_utils import mazes
from policy import Policy
import global_defs
from tqdm import tqdm

generator = mazes.KruskalMaze(n_x=global_defs.MAZE_HEIGHT, n_y=global_defs.MAZE_WIDTH)

maze = generator.maze
START = np.array([np.array(np.where(maze == 2)).flatten()])
END = np.array([np.array(np.where(maze == 3)).flatten()])
policy = Policy()


# one sample run = one 3D matrix
# one trajectory = one 2D matrix
# one row = one timestep
# -> timestep layout:
# [initial state, action, chosen actions' probability, new state, new reward, G_t]
@dataclass
class timestep:
    initial_state: np.ndarray
    action: int
    action_prob: float
    new_state: np.ndarray
    new_reward: float
    expected_reward: float

class Reinforce:
    def __init__(self, policy, start, end):
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


    def parse_maze_matrix(self, maze):
        self.maze = maze
        self.maze_mask = np.tile([0.0,0.0,0.0,0.0], (maze.shape[0], maze.shape[1], 1))
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i][j] == 1:
                    self.maze_mask[i][j] = [1, 1, 1, 1]
                    if((i-1) >= 0):
                        self.maze_mask[i-1][j][3] = 1
                    if((i+1) <= maze.shape[0]-1):
                        self.maze_mask[i+1][j][2] = 1
                    if((j-1) >= 0):
                        self.maze_mask[i][j-1][0] = 1
                    if((j+1) <= maze.shape[1]-1):
                        self.maze_mask[i][j+1][1] = 1

    def mask_policy_matrix(self, maze):
        self.policy_matrix = np.copy(self.raw_policy_matrix)
        print(self.policy_matrix.shape)
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                mask = self.maze_mask[i][j]
                idx = 0
                for int in mask:
                    if (int == 1):
                        self.policy_matrix[i][j][idx] = -1e11
                    idx += 1

    def populate_policy_matrix(self, maze):
        self.raw_policy_matrix = np.tile([0.0,0.0,0.0,0.0], (maze.shape[0], maze.shape[1], 1))
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                self.raw_policy_matrix[i][j] = self.policy.forward(np.array([[i, j]]))

    def prepare_policy_mask(self, maze):
        self.parse_maze_matrix(maze)
        self.populate_policy_matrix(maze)
        self.mask_policy_matrix(maze)
        self.policy_matrix = tf.nn.softmax(self.policy_matrix, axis=-1)

    def decide(self, probabilities):
        index = np.random.choice(len(global_defs.ACTIONS), p=probabilities)
        return global_defs.ACTIONS[index], probabilities[index].numpy()

    def get_action_for_state(self, local_state):
        state_x = local_state[0][0]
        state_y = local_state[0][1]
        probabilities = self.policy_matrix[state_x, state_y]
        action, action_prob = self.decide(probabilities)
        return action, action_prob

    def get_reward_for_new_state(self, new_state):
        if (np.array_equal(new_state, self.maze_end)):
            return 5.0
        return -1.0

    def timestep(self):
        current_state = self.state
        action, action_prob = self.get_action_for_state(current_state)

        new_state = current_state + action
        reward = self.get_reward_for_new_state(new_state)

        self.state = new_state

        t = timestep(
            current_state,
            action,
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

    def sample_run(self):
        if(self.need_clear_trajectories):
            self.trajectories = []
            self.need_clear_trajectories = False

        for i in tqdm(range(global_defs.SAMPLING_QUANTITY)):
            self.trajectory()

        self.trajectory_count = 0
        self.need_clear_trajectories = True


reinforce = Reinforce(policy, START, END)
reinforce.prepare_policy_mask(maze)
reinforce.sample_run()
print(reinforce.trajectories)
