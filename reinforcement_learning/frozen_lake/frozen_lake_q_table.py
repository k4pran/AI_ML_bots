import gym
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

torch.set_default_tensor_type('torch.DoubleTensor')

env = gym.make("FrozenLake-v0")
action_space = env.action_space.n
state_space = env.observation_space.n
holes = {5, 7, 11, 12}
reward_scale_factor = 1


class InverseModel(torch.nn.Module):

    def __init__(self, feature_space):
        super(InverseModel, self).__init__()
        self.double()

        self.feature_space = feature_space

        self.input = torch.nn.Linear(state_space, self.feature_space)
        self.out = torch.nn.Linear(self.feature_space * 2, action_space)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        # Catch the feature representations to be used by the forward model
        self.state_feature_rep = torch.zeros(self.feature_space)
        self.next_state_feature_rep = torch.zeros(self.feature_space)

    def get_state_features(self):
        return self.state_feature_rep.clone().detach()

    def get_next_state_features(self):
        return self.next_state_feature_rep.clone().detach()

    def forward(self, state, action, next_state):
        self.state_feature_rep = self.input(state)
        self.next_state_feature_rep = self.input(next_state)

        combined_feature_rep = torch.cat((self.state_feature_rep, self.next_state_feature_rep)).flatten()
        torch.nn.functional.elu(combined_feature_rep, inplace=True)

        pred_actions = self.out(combined_feature_rep)
        pred_actions = torch.nn.functional.softmax(pred_actions, dim=0)

        # Find loss
        loss = self.loss_fn(pred_actions.reshape(1, -1), torch.tensor([action]))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ForwardModel(torch.nn.Module):

    def __init__(self, feature_space):
        super(ForwardModel, self).__init__()

        self.feature_space = feature_space

        self.input = torch.nn.Linear(action_space + feature_space, 128)
        self.hidden = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, self.feature_space)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        self.double()

    def forward(self, action, state, next_state):
        combined_input = torch.cat((action, state), dim=1).flatten()
        x = self.input(combined_input)
        torch.nn.functional.elu(x, inplace=True)

        x = self.hidden(x)
        torch.nn.functional.elu(x, inplace=True)

        next_state_pred = self.output(x)

        loss = self.loss_fn(next_state_pred, next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class Agent(torch.nn.Module):

    def __init__(self, learning_rate=0.01, gamma=0.99, feature_space=32, verbose=False):
        super(Agent, self).__init__()

        self.feature_space = feature_space

        self.inverse_model = InverseModel(feature_space=feature_space)
        self.forward_model = ForwardModel(feature_space=feature_space)

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

        self.inverse_model.forward(state, torch.argmax(action), next_state)
        reward += self.forward_model.forward(action, self.inverse_model.get_state_features(),
                                           self.inverse_model.get_next_state_features())

        # Convert back to scalar
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

            next_state = np.array(next_state)
            next_state_encoded = OneHotEncoder(categories=[range(state_space)]).fit_transform(next_state.reshape(1, -1)).toarray()
            action_encoded = OneHotEncoder(categories=[range(action_space)]).fit_transform(np.array(action).reshape(1, -1)).toarray()

            agent.learn(torch.from_numpy(state_encoded), torch.from_numpy(action_encoded), reward,
                        torch.from_numpy(next_state_encoded), done)
            state = next_state

if __name__ == "__main__":
    frozen_agent = Agent(verbose=False)
    play(frozen_agent, render=True)
