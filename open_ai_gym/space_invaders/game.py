import gym
import numpy as np
from open_ai_gym.space_invaders.dqn_agent import DQNAgent

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    agent = DQNAgent(gamma=0.95, epsilon=1.0, learning_rate=0.003, memory_max=5000, epsilon_min=0.01, replace_target_count=None)

    while (agent.memory_count < agent.memory_max):

        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            if (done and info['ale.lives'] == 0):
                reward = -100

            agent.memorize(np.mean(state[15: 200, 30:125], axis=2), action, reward,
                           np.mean(next_state[15: 200, 30:125], axis=2))
    print("Done initializing memory")

    scores = []
    epsilon_history = []
    episodes = 50
    batch_size = 32

    for episode in range(episodes):
        print("starting game {} epsilon: {}".format(episode + 1, agent.epsilon))
        epsilon_history.append(agent.epsilon)
        done = False
        state = env.reset()
        frames = [np.sum(state[15: 200, 30:125], axis=2, dtype='float')]
        score = 0
        last_action = 0

        while not done:
            if len(frames) == 3:
                action = agent.choose_action(frames)
                frames = []
            else:
                action = last_action

            next_state, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(next_state[15: 200, 30:125], axis=2))
            if (done and info['ale.lives'] == 0):
                reward = -100

            agent.memorize(np.mean(state[15: 200, 30:125], axis=2), action, reward,
                           np.mean(next_state[15: 200, 30:125], axis=2))
            state = next_state
            agent.learn(batch_size)
            last_action = action
            # env.render()

        scores.append(score)
        print("score: {}".format(score))
        x = [i + 1 for i in range(episodes)]
        filename = "test" + str(episodes) + ".png"