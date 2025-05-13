import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import torch
import torch.nn.functional as F

# class DQN(nn.Module):
#     def __init__(self, input_dim=788, output_dim=5):
#         super(DQN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )

#     def forward(self, x):
#         return self.net(x)

class DQN(nn.Module):
    def __init__(self, output_dim=5):
        super().__init__()
        self.image_conv = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # keeps 28x28
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # down to 14x14
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # down to 7x7
        nn.ReLU(),
)
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 4, 128),  # match your original structure (extra 4 features)
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state):
        image = state[:, :784].view(-1, 1, 28, 28)  # reshape flat image
        meta = state[:, 784:]  # [x, y, predict_count, reveal%]
        x = self.image_conv(image)
        x = x.view(x.size(0), -1)  # flatten conv output
        x = torch.cat((x, meta), dim=1)
        return self.fc(x)




class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)



class DQNAgent:
    def __init__(self, state_dim=788, n_actions=5, device="cpu", gamma=0.99, lr=1e-4):
        self.device = device
        # self.q_network = DQN(state_dim, n_actions).to(device)
        # self.target_network = DQN(state_dim, n_actions).to(device)
        self.q_network = DQN(output_dim=n_actions).to(device)
        self.target_network = DQN(output_dim=n_actions).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.total_steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.998
        self.n_actions = n_actions

    def select_action(self, state):
        self.total_steps += 1
        if self.total_steps < 100:
            return np.random.randint(self.n_actions)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # states = states.to(self.device)
        # next_states = next_states.to(self.device)
        states = torch.from_numpy(np.array(states)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


        # next_q_values = self.target_network(next_states)
        #for double dqn
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_value, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # q_target = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
        

        # loss = F.mse_loss(q_value, q_target.detach())
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # print(f"Q-values (sample): {q_values[0].detach().cpu().numpy()}")



        # Epsilon decay
        if self.epsilon > self.epsilon_min :
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
