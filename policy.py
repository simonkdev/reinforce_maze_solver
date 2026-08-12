import tensorflow as tf
import numpy as np
import global_defs

class Policy:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(len(global_defs.ACTIONS))
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#
    def forward(self, state):
        tensor = self.model(state)
        predictions = tensor.numpy()
        return predictions
