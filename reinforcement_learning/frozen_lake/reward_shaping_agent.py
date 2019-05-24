import gym
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

torch.set_default_tensor_type('torch.DoubleTensor')

env = gym.make("FrozenLake-v0")
action_space = env.action_space.n
state_space = env.observation_space.n
holes = {5, 7, 11, 12}


class Agent(torch.nn.Module):

    def __init__(self, learning_rate=0.005, gamma=0.99, verbose=False):
        super(Agent, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_space, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, action_space),
            torch.nn.Softmax(1)
        )

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        self.training_count = 0
        self.verbose = verbose

        self.learning_rate = learning_rate
        self.gamma = gamma

    def act(self, state):
       return int(torch.argmax(self.model(state)).item())

    def learn(self, state, action, reward, next_state, done):

        action = torch.argmax(action)

        current_pred = self.model(state)[0][action]
        max_future_pred = torch.max(self.model(next_state))
        q_target = reward + (self.gamma * max_future_pred)

        loss = self.loss_fn(current_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def play(agent, episodes=10000, render=False):

    for episode in range(episodes):

        state = env.reset()
        done = False
        while not done:

            if render:
                env.render()

            state_encoded = OneHotEncoder(categories=[range(state_space)]).fit_transform(state.reshape(1, -1)).toarray()
            action = agent.act(torch.from_numpy(state_encoded))
            next_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -10

            elif reward == 0:
                reward = -1

            else:
                reward = 10
                print("GOAL at {}".format(episode))

            next_state = np.array(next_state)

            next_state_encoded = OneHotEncoder(categories=[range(state_space)]).fit_transform(next_state.reshape(1, -1)).toarray()
            action_encoded = OneHotEncoder(categories=[range(action_space)]).fit_transform(np.array(action).reshape(1, -1)).toarray()

            agent.learn(torch.from_numpy(state_encoded), torch.from_numpy(action_encoded), reward,
                        torch.from_numpy(next_state_encoded), done)
            state = next_state

if __name__ == "__main__":
    frozen_agent = Agent(verbose=False)
    play(frozen_agent, render=False)