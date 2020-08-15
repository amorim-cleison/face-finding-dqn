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
        # cur_state = env.reset().reshape(1, 2)
        cur_state = env.reset()

        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action)
            env.render()
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
rnd_pointrnd_pointrnd_pointpointrnd_norm_pointrnd_norm_pointrnd_norm_pointnorm_pointrnd_centerrnd_centerrnd_centermin_centermin_centermax_centermax_centercentercenterscalenew_statenew_statenew_viewnew_viewnew_view