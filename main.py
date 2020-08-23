from model import DQN
from environment import PeopleFramingEnv

dqn_params = {
    "minibatch_size": 32,
    "replay_memory_size": 1000000,
    "target_net_update_freq": 10000,
    "discount_factor": 0.99,
    "learning_rate": 0.00025,
    "gradient_momentum": 0.95,  # RMSProp

    "initial_epsilon": 1.,
    "final_epsilon": 0.1,
    "final_exploration_frame": 1000000,

    "num_episodes": 100000,
    "max_episode_len": 10,

    "save_dir": "./tmp/",
    "checkpoint_file": "./tmp/LAST.pkl"
}


env_params = {"img_path": "data/image/sl-person-003.png"}


def run(env_params, dqn_params):
    env = PeopleFramingEnv(**env_params)
    dqn_agent = DQN(env=env, **dqn_params)
    dqn_agent.run()


def print_start(episode):
    print("=" * 60)
    print(f"EPISODE {(episode+1)}")
    print("-" * 60)
    print(f"{'Step':5} | {'State':<25}\t {'Action'}\t {'Reward'}")
    print("-" * 60)


def print_progress(step, state, action, reward):
    print(
        f"{(step+1):<5} | S: {hash(str(state)):<25}\t A: {action}\t R: {reward:.2f}"
    )


def print_end(success, steps):
    print("-" * 60)
    if success:
        print(f" -> Completed in {(steps+1)} episodes")
    else:
        print(" -> Failed to complete trial")
    print()


if __name__ == "__main__":
    run(env_params, dqn_params)
