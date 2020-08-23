import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, Flatten
from keras.optimizers import RMSprop

from collections import deque
from .logger import log_start, log_progress, log_finish
from os.path import normpath, exists
from os import mkdir
import pickle


class DQN:
    """
    TODO: document DQN
    """
    def __init__(self,
                 env,
                 minibatch_size,
                 replay_memory_size,
                 target_net_update_freq,
                 discount_factor,
                 learning_rate,
                 gradient_momentum,
                 initial_epsilon,
                 final_epsilon,
                 final_exploration_frame,
                 num_episodes,
                 max_episode_len,
                 save_dir,
                 checkpoint_file=None):
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = (initial_epsilon -
                              final_epsilon) / final_exploration_frame

        self.env = env
        self.learning_rate = learning_rate
        self.gradient_momentum = gradient_momentum
        self.minibatch_size = minibatch_size
        self.save_dir = normpath(save_dir)
        self.checkpoint_file = normpath(checkpoint_file)

        self.discount_factor = discount_factor
        self.m = num_episodes
        self.t = max_episode_len
        self.c = target_net_update_freq

        self.q_target_weights = None
        self.q_weights = None
        self.start_episode = 1

        # Restore checkpoint:
        self.__restore_checkpoint()

        # Initialize replay memory D to capacity N
        self.d = deque(maxlen=replay_memory_size)

        # Initialize action-value function Q with random weights
        # `thetha`
        self.q = self.__create_model(self.q_weights)

        # Initialize target action-value function ^Q with weights
        # `thetha' = thetha`
        self.q_target = self.__create_model(self.q_target_weights)

    def run(self):
        # For episode = 1, M do
        for episode in range(self.start_episode, self.m + 1):
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

            # Log result and save checkpoint:
            log_finish(finish, episode)
            self.__save_checkpoint(episode)

    def __preprocess(self, s, a, x_next):
        x_next = x_next[np.newaxis, :]

        if not x_next.flags['C_CONTIGUOUS']:
            x_next = np.ascontiguousarray(x_next)
        return (s, a, x_next)

    def __create_model(self, weights=None):
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
                      optimizer=RMSprop(learning_rate=self.learning_rate,
                                        momentum=self.gradient_momentum))

        self.__load_weights(model, weights)
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

    def __save_checkpoint(self, episode):
        checkpoint = dict()
        print(f"Saving checkpoint (ep{episode})...")

        def __create_dir(path):
            if not exists(path):
                mkdir(path)

        def __save_weights(model, name, episode):
            # path = normpath(f"{weights_dir}/ep{episode}-{name}")
            path = normpath(f"{weights_dir}/LAST-{name}")
            model.save_weights(path, overwrite=True, save_format="h5")
            return path

        def __save_ckp(data, name):
            path = normpath(f"{ckp_dir}/{name}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Directories:
        ckp_dir = normpath(f"{self.save_dir}")
        weights_dir = normpath(f"{ckp_dir}/weights")
        __create_dir(ckp_dir)
        __create_dir(weights_dir)

        # Parameters:
        checkpoint["q_weights"] = __save_weights(self.q, "q", episode)
        checkpoint["q_target_weights"] = __save_weights(
            self.q_target, "q_target", episode)
        checkpoint["episode"] = episode

        # Save:
        # __save_ckp(checkpoint, f"ep{episode}")
        __save_ckp(checkpoint, "LAST")

    def __restore_checkpoint(self):
        # TODO: assert file existence or skip?
        if self.checkpoint_file is not None and exists(self.checkpoint_file):
            print("Loading checkpoint...")

            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

                assert isinstance(checkpoint, dict), "Error loading checkpoint"
                self.q_weights = checkpoint["q_weights"]
                self.q_target_weights = checkpoint["q_target_weights"]
                self.start_episode = checkpoint["episode"] + 1

    def __load_weights(self, model, weights):
        if weights is not None:
            print(f"Loading '{weights}'...")
            model.load_weights(weights)
