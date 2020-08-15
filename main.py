from model import DQN
from env import PeopleFramingEnv

dqn_params = {
    "minibatch_size": 32,
    "replay_memory_size": 1000000,
    "target_net_update_freq": 10000,
    "discount_factor": 0.99,
    "learning_rate": 0.00025,
    "gradient_momentum": 0.95,  # RMSProp
    "squared_gradient_momentum": 0.95,  # RMSProp
    "min_gradient_momentum": 0.01,  # RMSProp

    "initial_epsilon": 1.,
    "final_epsilon": 0.1,
    "final_exploration_frame": 1000000,

    # "replay_start_size": 50000,

    # "epsilon_decay": 0.995,
    "num_episodes": 100000,
    "max_episode_len": 20,
}


env_params = {"img_path": "data/image/sl-person-003.png"}


def run(env_params, dqn_params):
    env = PeopleFramingEnv(**env_params)
    dqn_agent = DQN(env=env, **dqn_params)
    dqn_agent.run()


    # for episode in range(episodes):
    #     cur_state = env.reset()
    #     print_start(episode)

    #     for step in range(max_episode_size):
    #         action = dqn_agent.__select_action(cur_state)
    #         new_state, reward, done = env.step(action)
    #         env.render()
    #         print_progress(step, cur_state, action, reward)
    #         dqn_agent.remember(cur_state, action, reward, new_state, done)

    #         dqn_agent.__replay()
    #         dqn_agent.__target_train()
    #         cur_state = new_state
    #         if done:
    #             break

    #     # Print result:
    #     success = (step < (max_episode_size - 1))
    #     print_end(success, episode)


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
