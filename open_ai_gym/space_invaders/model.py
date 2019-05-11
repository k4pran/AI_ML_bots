import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ALPHA = 0.01


class DQNNetwork(nn.Module):

    def __init__(self, learning_rate):
        super(DQNNetwork, self).__init__()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dense1 = nn.Linear(128 * 19 * 8, 512)
        self.dense2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        # observation = observation.view(-1, 3, 210, 160).to(self.device)
        observation = observation.view(-1, 1, 185, 95)
        # [210, 160, 3]

        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        # observation = observation.view(-1, 128 * 23 * 16).to(self.device)
        observation = observation.view(-1, 128 * 19 * 8)

        observation = F.relu(self.dense1(observation))

        actions = self.dense2(observation)
        return actions
