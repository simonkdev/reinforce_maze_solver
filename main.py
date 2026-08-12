from dataclasses import dataclass

import tensorflow as tf
from tensorflow.dtypes import float32
import numpy as np
from maze_utils import mazes
from policy import Policy
import global_defs

generator = mazes.KruskalMaze(n_x=global_defs.MAZE_HEIGHT, n_y=global_defs.MAZE_WIDTH)

maze = generator.maze
start_point = np.array([np.array(np.where(maze == 2)).flatten()])
end_point = np.array([np.array(np.where(maze == 3)).flatten()])
state = start_point
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

@dataclass
class trajectory:
    timesteps: list

@dataclass
class sample_run:
    trajectories: list


class Reinforce:
    def __init__(self, policy):
         self.hello = "Hello World"
         self.policy = policy
         self.terminate_trajectory = False

    def parse_maze_matrix(self, maze):
        self.maze = maze
        self.maze_mask = np.tile([0.0,0.0,0.0,0.0], (maze.shape[0], maze.shape[1], 1))
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i][j] == 1:
                    self.maze_mask[i][j] = [1, 1, 1, 1]
                    if((i-1) >= 0):
                        self.maze_mask[i-1][j][2] = 1
                    if((i+1) <= maze.shape[0]-1):
                        self.maze_mask[i+1][j][3] = 1
                    if((j-1) >= 0):
                        self.maze_mask[i][j-1][1] = 1
                    if((j+1) <= maze.shape[1]-1):
                        self.maze_mask[i][j+1][0] = 1

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

    def test_masking(self, maze):
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                mask = self.maze_mask[i, j]
                probs = self.policy_matrix[i, j].numpy()

                for action in range(4):
                    if mask[action] == 1 and probs[action] > 1e-6:
                        print("ERROR:", i, j, action, probs[action])

    # def step(self):
    #     probabilities = self.policy.forward(state)
    #     action, action_prob = self.decide(probabilities)
    #     ini_state = state
    #     self.move_agent(action)
    #     reward = self.get_reward_value()
    #     t = timestep(ini_state, action, action_prob, state, reward, 0.0)

    # def run_trajectories(self):
    #     print("")

    # def new_sample_run(self):
    #     self.sample_run = sample_run()

    # def decide(self, probabilities):
    #     index = np.random.choice(len(global_defs.ACTIONS), p=probabilities)
    #     return global_defs.ACTIONS[index], probabilities[index]

    # def move_agent(self, step):
    #     print("")

    # def get_reward_value(self):
    #     if(state == end_point):
    #         return 3
    #     if(self.check_if_blocked(state)):
    #         self.terminate_trajectory = True
    #     return 0

    # def check_if_blocked(self, target_state):
    #     if(maze[target_state[0][0]][target_state[0][1]] == 1):
    #         return True
    #     return False


reinforce = Reinforce(policy)
reinforce.prepare_policy_mask(maze)
print("MAZE: ")
print(maze)
print("RAW POLICY MATRIX: ")
print(reinforce.raw_policy_matrix)
print("MAZE MASK: ")
print(reinforce.maze_mask)
print("MASKED POLICY MATRIX: ")
print(reinforce.policy_matrix)
#reinforce.test_masking(maze)
