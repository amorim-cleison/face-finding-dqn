import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, Flatten
from keras.optimizers import RMSprop

from collections import deque
from .logger import log_start, log_progress, log_finish, log_model, log_args
from os.path import normpath, exists
from os import mkdir
import pickle
import csv
from datetime import datetime
from hashlib import md5
from environment.image_utils import resize_raw, get_luminance


class DQN:
    """
    TODO: document DQN
    """
    def __init__(self, checkpoint_file=None, **kwargs):
        # Restore checkpoint:
        restored_args = self.__restore_checkpoint(checkpoint_file)
        args = restored_args if restored_args is not None else kwargs
        log_args(args)
        self.__init_dqn(checkpoint_file=checkpoint_file, **args)

    def __init_dqn(self,
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
                   stacked_frames,
                   save_dir,
                   checkpoint_file=None,
                   state_shape=(84, 84),
                   d=None,
                   stacked_frames_d=None,
                   q_weights=None,
                   q_target_weights=None,
                   last_episode=0,
                   epsilon=None,
                   **kwargs):
        self.epsilon = initial_epsilon if epsilon is None else epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_exploration_frame = final_exploration_frame
        self.epsilon_decay = (initial_epsilon -
                              final_epsilon) / final_exploration_frame

        self.env = env
        self.start_episode = last_episode + 1
        self.learning_rate = learning_rate
        self.gradient_momentum = gradient_momentum
        self.minibatch_size = minibatch_size
        self.save_dir = normpath(save_dir)
        self.checkpoint_file = normpath(checkpoint_file)
        self.state_shape = state_shape

        self.discount_factor = discount_factor
        self.m = num_episodes
        self.t = max_episode_len
        self.c = target_net_update_freq

        self.stacked_frames = stacked_frames
        self.stacked_frames_d = self.__init_stacked_frames(
            stacked_frames,
            state_shape) if stacked_frames_d is None else stacked_frames_d

        # Initialize replay memory D to capacity N
        self.replay_memory_size = replay_memory_size
        self.d = deque(maxlen=replay_memory_size) if d is None else d

        # Initialize action-value function Q with random weights
        # `thetha`
        self.q = self.__create_model(q_weights)
        log_model(self.q)

        # Initialize target action-value function ^Q with weights
        # `thetha' = thetha`
        self.q_target = self.__create_model(q_target_weights)

    def run(self):
        # For episode = 1, M do
        for episode in range(self.start_episode, self.m + 1):
            logs = []

            # Initialize sequence `s_1 = {x_1}` and preprocessed
            # sequence `fi_1 = fi(s_1)`:
            x = self.env.reset()
            s = (None, None, x)
            fi = self.__preprocess(*s)

            log_start(episode)
            self.env.render()

            for t in range(1, self.t + 1):
                # Select action:
                a, a_name = self.__select_action(fi)

                # Execute action `a_t` in emulator and observe reward `r_t`
                # and image `x_t+1`:
                x_next, r, finish = self.env.step(a)

                # Set `s_t+1 = s_t, a_t, x_t+1` and preprocess
                # `fi_t+1 = fi(s_t+1)`:
                s_next = (s, a, x_next)
                fi_next = self.__preprocess(*s_next)

                # Store transition (fi_t, a_t, r_t, fi_t+1) in D:
                self.d.append((fi, a, r, fi_next, finish))

                # Replay from memory:
                loss, accuracy = self.__replay()
                log_info = {
                    "datetime": datetime.now(),
                    "episode": episode,
                    "step": t,
                    "state": self.__get_hash(fi),
                    "action": a,
                    "action_name": a_name,
                    "reward": r,
                    "loss": loss,
                    "accuracy": accuracy,
                    "finish": finish
                }
                self.env.render()
                log_progress(**log_info)
                logs.append(log_info)

                # Every C steps reset `^Q = Q`:
                if (t % self.c == 0):
                    self.__update_target_network()

                # Update current values:
                x, fi, s = x_next, fi_next, s_next

                if finish:
                    break

            # Log result and save checkpoint:
            log_finish(finish, t)
            self.__save_checkpoint(episode)
            self.__write_logs(logs)

    def __init_stacked_frames(self, size, state_shape):
        return deque(np.zeros((size, *state_shape)), maxlen=size)

    def __preprocess(self, s, a, x_next):
        def process(x):
            x_proc = x
            x_proc = resize_raw(x_proc, *self.state_shape)
            x_proc = get_luminance(x_proc)
            return x_proc

        def to_contiguous(x):
            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x)
            return x

        def reorganize(x):
            return np.transpose(x, (1, 2, 0))

        x_next_proc = process(x_next)
        self.stacked_frames_d.append(x_next_proc)
        return to_contiguous(reorganize(self.stacked_frames_d))

    def __create_model(self, weights):
        output_units = self.env.action_space.n

        layers = [
            InputLayer(name="input",
                       input_shape=(*self.state_shape,
                                    len(self.stacked_frames_d))),
            Conv2D(name="conv_1",
                   filters=32,
                   kernel_size=(8, 8),
                   strides=4,
                   activation="relu"),
            Conv2D(name="conv_2",
                   filters=64,
                   kernel_size=(4, 4),
                   strides=2,
                   activation="relu"),
            Conv2D(name="conv_3",
                   filters=64,
                   kernel_size=(3, 3),
                   strides=1,
                   activation="relu"),
            Flatten(name="flatten"),
            Dense(name="dense_1", units=512, activation="relu"),
            Dense(name="dense_2", units=output_units)
        ]

        model = Sequential(layers, "DQN")
        model.compile(loss="mean_squared_error",
                      metrics=['accuracy', 'mse'],
                      optimizer=RMSprop(learning_rate=self.learning_rate,
                                        momentum=self.gradient_momentum))

        self.__load_weights(model, weights)
        return model

    def __select_action(self, fi):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.final_epsilon)

        # With probability `epsilon` select a random action `at`
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # otherwise select `at = argmaxa Q(fi(st), a; thetha)`
            action = np.argmax(self.q.predict(self.__to_single_batch(fi)))
        return action, self.env.get_action_name(action)

    def __replay(self):
        # Sample random minibatch of transitions (fi_j, a_j, r_j, fi_j+1) from D:
        size = min(len(self.d), self.minibatch_size)
        minibatch = random.sample(self.d, size)
        losses = []
        accuracies = []

        for (fi, a, r, fi_next, finish) in minibatch:
            if finish:
                # Set `y_j = r_j` if episode terminates at step `j+1`:
                y = r
            else:
                # Otherwise, set `y_j = gamma * max_a'(^Q(fi_j+1, a'; theta')`
                y = r + (self.discount_factor * np.max(
                    self.q_target.predict(self.__to_single_batch(fi_next))))

            target = self.q_target.predict(self.__to_single_batch(fi))
            target[0][a] = y

            # Perform a gradient descent step on
            # `(y_j - Q(fi_j, a_j; thetha))^2` with respect to the network
            # parameters `thetha`:
            output = self.q.fit(self.__to_single_batch(fi_next),
                                target,
                                epochs=1,
                                batch_size=1,
                                verbose=0)
            losses.append(output.history["loss"][0])
            accuracies.append(output.history["accuracy"][0])
        return np.average(losses), np.average(accuracies)

    def __update_target_network(self):
        weights = self.q.get_weights()
        target_weights = self.q_target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.q_target.set_weights(target_weights)

    def __save_checkpoint(self, episode):
        checkpoint = dict()
        print("Saving checkpoint...")

        def __create_dir(path):
            if not exists(path):
                mkdir(path)

        def __save_weights(model, name, episode):
            # path = normpath(f"{weights_dir}/ep{episode}-{name}")
            path = normpath(f"{weights_dir}/LAST-{name}.h5")
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

        checkpoint = {
            "env": self.env,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "final_exploration_frame": self.final_exploration_frame,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "gradient_momentum": self.gradient_momentum,
            "minibatch_size": self.minibatch_size,
            "replay_memory_size": self.replay_memory_size,
            "save_dir": self.save_dir,
            "state_shape": self.state_shape,
            "discount_factor": self.discount_factor,
            "num_episodes": self.m,
            "max_episode_len": self.t,
            "stacked_frames": self.stacked_frames,
            "target_net_update_freq": self.c,
            "d": self.d,
            "stacked_frames_d": self.stacked_frames_d,
            "q_weights": __save_weights(self.q, "q", episode),
            "q_target_weights": __save_weights(self.q_target, "q_target",
                                               episode),
            "last_episode": episode
        }

        # Save:
        __save_ckp(checkpoint, "LAST")

    def __restore_checkpoint(self, path):
        if path is not None and exists(path):
            print("Loading checkpoint...")

            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
                return checkpoint
        return None

    def __load_weights(self, model, weights):
        if weights is not None:
            print(f"Loading '{weights}'...")
            model.load_weights(weights)

    def __write_logs(self, logs):
        path = normpath(f"{self.save_dir}/log.csv")
        write_header = not exists(path)

        if len(logs) > 0:
            with open(path, 'a', newline='') as csvfile:
                columns = logs[0].keys()
                writer = csv.DictWriter(csvfile,
                                        fieldnames=columns,
                                        delimiter=";")

                if write_header:
                    writer.writeheader()
                writer.writerows(logs)

    def __get_hash(self, state):
        return md5(state).hexdigest()

    def __to_single_batch(self, x):
        return x[np.newaxis, :]
