import torch
import numpy as np
from open_ai_gym.space_invaders.model import DQNNetwork

torch.set_default_tensor_type('torch.DoubleTensor')

class DQNAgent:

    def __init__(self, gamma, epsilon, learning_rate, memory_max, epsilon_min, replace_target_count=10000,
                 action_space=[i for i in range(1, 6, 1)]):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.memory_max = memory_max
        self.action_space = action_space

        self.steps = 0
        self.learning_step_counter = 0
        self.memory = []
        self.memory_count = 0
        self.replace_target_count = replace_target_count
        self.q_eval = DQNNetwork(learning_rate)
        self.q_next = DQNNetwork(learning_rate)

    def memorize(self, state, action, reward, next_state):
        if (self.memory_count < self.memory_max):
            self.memory.append([state, action, reward, next_state])
        else:
            self.memory[self.memory_count % self.memory_max] = [state, action, reward, next_state]
        self.memory_count += 1

    def choose_action(self, observation):
        rand = np.random.random()
        actions = self.q_eval.forward(observation)

        if rand < 1 - self.epsilon:
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.q_eval.optimizer.zero_grad()
        if (self.replace_target_count is not None and
                self.learning_step_counter % self.replace_target_count == 0):
            self.q_next.load_state_dict(self.q_eval.state_dict())

        if (self.memory_count + batch_size < self.memory_max):
            mem_start = int(np.random.choice(range(self.memory_count)))
        else:
            mem_start = int(np.random.choice(range(self.memory_max - batch_size - 1)))

        mini_batch = self.memory[mem_start: mem_start + batch_size]
        memory = np.array(mini_batch)

        q_pred = self.q_eval.forward(list(memory[:, 0][:])).to(self.q_eval.device)
        q_next = self.q_eval.forward(list(memory[:, 3][:])).to(self.q_eval.device)

        max_action = torch.argmax(q_next, dim=1).to(self.q_eval.device)
        rewards = torch.tensor(list(memory[:, 2])).to(self.q_eval.device)
        q_target = q_pred
        q_target[:, max_action] = rewards + self.gamma * torch.max(q_next[1])

        if (self.steps > 500):
            if (self.epsilon - 1e-4 > self.epsilon_min):
                self.epsilon -= 1e-4
            else:
                self.epsilon = self.epsilon_min

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learning_step_counter += 1