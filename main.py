from dqn import DQN
from env import PeopleFramingEnvironment


def main():
    # env = gym.make("MountainCar-v0")
    env = PeopleFramingEnvironment("data/image/sl-person-003.png", False)

    trials = 100
    # trial_len = 500
    trial_len = 20
    dqn_agent = DQN(env=env)

    for trial in range(trials):
        print_start(trial)

        cur_state = env.reset()

        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action)
            env.render()
            # reward = reward if not done else -20
            print_progress(step, cur_state, action, reward)
            # new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break
        if step >= (trial_len - 1):
            print_end(False, trials)
        else:
            print_end(True, trials)


def print_start(trial):
    print("=" * 60)
    print(f"TRIAL {(trial+1)}")
    print("-" * 60)
    print(f"{'Step':5} | {'State':<25}\t {'Action'}\t {'Reward'}")
    print("-" * 60)


def print_progress(step, state, action, reward):
    print(
        f"{(step+1):<5} | S: {hash(str(state)):<25}\t A: {action}\t R: {reward:.2f}"
    )


def print_end(success, trials):
    print("-" * 60)
    if success:
        print(f" -> Completed in {trials} trials")
    else:
        print(" -> Failed to complete trial")
    print()


if __name__ == "__main__":
    main()
