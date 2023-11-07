import logging

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate, discount_factor, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_network = self.build_q_network()
        self.target_q_network = self.build_q_network()

    def build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(self.state_space,), activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values)

    def update_q_values(self, state, action, reward, next_state):
        target_q_values = self.target_q_network.predict(state.reshape(1, -1))
        next_q_values = self.q_network.predict(next_state.reshape(1, -1))
        target_q_values[0][action] = reward + self.discount_factor * np.max(next_q_values)
        self.q_network.fit(state.reshape(1, -1), target_q_values, epochs=1, verbose=0)
