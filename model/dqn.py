import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop

from collections import deque
from .logger import log_start, log_progress, log_finish


class DQN:
    """
    DQN implementation as per
    https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    """
    def __init__(
        self,
        env,
        minibatch_size,
        replay_memory_size,
        target_net_update_freq,
        discount_factor,
        learning_rate,

        gradient_momentum,
        squared_gradient_momentum,
        min_gradient_momentum,

        initial_epsilon,
        final_epsilon,
        final_exploration_frame,

        num_episodes,
        max_episode_len,
    ):
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = (initial_epsilon -
                              final_epsilon) / final_exploration_frame

        self.env = env
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size

        self.discount_factor = discount_factor
        self.m = num_episodes
        self.t = max_episode_len
        self.c = target_net_update_freq

        # Initialize replay memory D to capacity N
        self.d = deque(maxlen=replay_memory_size)

        # Initialize action-value function Q with random weights
        # `thetha`
        self.q = self.__create_model()

        # Initialize target action-value function ^Q with weights
        # `thetha' = thetha`
        self.q_target = self.__create_model()

    def run(self):
        # For episode = 1, M do
        for episode in range(1, self.m + 1):
            # Initialize sequence `s_1 = {x_1}` and preprocessed
            # sequence `fi_1 = fi(s_1)`:
            x = self.env.reset()
            s = (None, None, x)
            fi = self.__preprocess(*s)

            log_start(episode)

            for t in range(1, self.t + 1):
                # Select action:
                a = self.__select_action(fi)

                # Execute action `a_t` in emulator and observe reward `r_t`
                # and image `x_t+1`:
                x_next, r, finish = self.env.step(a)
                self.env.render()
                log_progress(t, fi[2], a, r)

                # Set `s_t+1 = s_t, a_t, x_t+1` and preprocess
                # `fi_t+1 = fi(s_t+1)`:
                s_next = (s, a, x_next)
                fi_next = self.__preprocess(*s_next)

                # Store transition (fi_t, a_t, r_t, fi_t+1) in D:
                self.d.append((fi, a, r, fi_next, finish))

                # Replay from memory:
                self.__replay()

                # Every C steps reset `^Q = Q`:
                if (t % self.c == 0):
                    self.__update_target_network()

                # Update current values:
                x, fi, s = x_next, fi_next, s_next

                if finish:
                    break

            # Log result:
            log_finish(finish, episode)

    def __preprocess(self, s, a, x_next):
        # TODO: implementar
        x_next = x_next[np.newaxis, :]

        if not x_next.flags['C_CONTIGUOUS']:
            x_next = np.ascontiguousarray(x_next)
        return (s, a, x_next)

    def __create_model(self):
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

        # FIXME: RMSprop

        return model

    def __select_action(self, fi):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.final_epsilon)

        # With probability `epsilon` select a random action `at`
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # otherwise select `at = argmaxa Q(fi(st), a; thetha)`
            return np.argmax(self.q.predict(fi[2]))

    def __replay(self):
        # Sample random minibatch of transitions (fi_j, a_j, r_j, fi_j+1) from D:
        size = min(len(self.d), self.minibatch_size)
        minibatch = random.sample(self.d, size)

        for (fi, a, r, fi_next, finish) in minibatch:
            if finish:
                # Set `y_j = r_j` if episode terminates at step `j+1`:
                y = r
            else:
                # Otherwise, set `y_j = gamma * max_a'(^Q(fi_j+1, a'; theta')`
                y = r + (self.discount_factor *
                         np.max(self.q_target.predict(fi_next[2])))

            target = self.q_target.predict(fi[2])
            target[0][a] = y

            # Perform a gradient descent step on
            # `(y_j - Q(fi_j, a_j; thetha))^2` with respect to the network
            # parameters `thetha`:
            self.q.fit(fi_next[2], target, epochs=1, verbose=0)

    def __update_target_network(self):
        weights = self.q.get_weights()
        target_weights = self.q_target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i]

        self.q_target.set_weights(target_weights)
