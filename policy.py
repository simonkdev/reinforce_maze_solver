import tensorflow as tf
import global_defs


class TabularPolicyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.logits = self.add_weight(
            shape=(global_defs.MAZE_HEIGHT + 2, global_defs.MAZE_WIDTH + 2, len(global_defs.ACTIONS)),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            name="cell_action_logits",
        )

    def call(self, state):
        state_indices = tf.cast(state, tf.int32)
        return tf.gather_nd(self.logits, state_indices)


class Policy:
    def __init__(self):
        self.model = TabularPolicyModel()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=global_defs.LEARNING_RATE)

    def forward(self, state):
        tensor = self.model(state)
        predictions = tensor.numpy()
        return predictions
