from model import DQN
from environment import PeopleFramingEnv

# dqn_params = {
#     "minibatch_size": 32,
#     "replay_memory_size": 1000000,
#     "target_net_update_freq": 10000,
#     "discount_factor": 0.99,
#     "learning_rate": 0.00025,
#     "gradient_momentum": 0.95,  # RMSProp

#     "initial_epsilon": 1.,
#     "final_epsilon": 0.1,
#     "final_exploration_frame": 1000000,

#     "num_episodes": 100000,
#     "max_episode_len": 10,

#     "save_dir": "./tmp/",
#     "checkpoint_file": "./tmp/LAST.pkl"
# }

env_params = {
    "img_path": "data/image/sl-person-001.png", 
    "draw_roi": False
}

dqn_params = {
    "minibatch_size": 20,
    "replay_memory_size": 1000,
    "target_net_update_freq": 15,
    "discount_factor": 0.99,
    "learning_rate": 0.00025,
    "gradient_momentum": 0.95,  # RMSProp
    "initial_epsilon": 1.,
    "final_epsilon": 0.1,
    "final_exploration_frame": 10000,
    "num_episodes": 100000,
    "max_episode_len": 30,
    "stacked_frames": 3,
    "save_dir": "./tmp/",
    "checkpoint_file": "./tmp/LAST.pkl"
}


def run(env_params, dqn_params):
    env = PeopleFramingEnv(**env_params)
    dqn_agent = DQN(env=env, **dqn_params)
    dqn_agent.run()


if __name__ == "__main__":
    run(env_params, dqn_params)
