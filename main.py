from dqn import DQN
from env import PeopleFramingEnv

dqn_params = {
    "memory_size": 2000,
    "gamma": 0.95,
    "epsilon_decay": 0.995,
    "learning_rate": 0.01,
    "replay_batch_size": 32
}

episodes = 100000
episode_len = 20

env_params = {"img_path": "data/image/sl-person-003.png"}


def run(episodes, max_episode_size, env_params, dqn_params):
    env = PeopleFramingEnv(**env_params)
    dqn_agent = DQN(env, **dqn_params)

    for episode in range(episodes):
        cur_state = env.reset()
        print_start(episode)

        for step in range(max_episode_size):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action)
            env.render()
            print_progress(step, cur_state, action, reward)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break

        # Print result:
        success = (step < (max_episode_size - 1))
        print_end(success, episode)


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


def print_end(success, episode):
    print("-" * 60)
    if success:
        print(f" -> Completed in {(episode+1)} episodes")
    else:
        print(" -> Failed to complete trial")
    print()


if __name__ == "__main__":
    run(episodes, episode_len, env_params, dqn_params)
