from dqn import DQN
from face_env import FaceEnvironment


def main():
    # env = gym.make("MountainCar-v0")
    env = FaceEnvironment("data/image/cityscape-person-600.jpg")

    trials = 100
    # trial_len = 500
    trial_len = 20
    dqn_agent = DQN(env=env)

    for trial in range(trials):
        # cur_state = env.reset().reshape(1, 2)
        cur_state = env.reset()

        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -20
            print(reward)
            # new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break
        if step >= (trial_len - 1):
            print("Failed to complete trial")
        else:
            print("Completed in {} trials".format(trial))
            # break


if __name__ == "__main__":
    main()
