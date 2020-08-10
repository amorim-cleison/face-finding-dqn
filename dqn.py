import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, Flatten
from keras.optimizers import Adam

from collections import deque


class DQN:
    """
    DQN implementation as per
    https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    """
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .05

        self.model = self.create_model()
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self.create_model()

    def create_model(self):
        state_shape = self.env.observation_space.shape
        output_units = self.env.action_space.n

        layers = [
            InputLayer(input_shape=state_shape),
            Conv2D(filters=32,
                   kernel_size=(8, 8),
                   strides=4,
                   activation="relu"),
            Conv2D(filters=64,
                   kernel_size=(4, 4),
                   strides=2,
                   activation="relu"),
            Conv2D(filters=64,
                   kernel_size=(3, 3),
                   strides=1,
                   activation="relu"),
            Flatten(),
            Dense(units=256, activation="relu"),
            Dense(units=output_units)
        ]

        model = Sequential(layers, "DQN")
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state[np.newaxis, ...]))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state[np.newaxis, ...])
            if done:
                target[0][action] = reward
            else:
                Q_future = max(
                    self.target_model.predict(new_state[np.newaxis, ...])[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state[np.newaxis, ...], target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
