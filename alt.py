#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import reveal_mnist
import gymnasium as gym

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
TARGET_UPDATE = 10
LEARNING_RATE = 0.0001
MEMORY_SIZE = 10000


class DQN(nn.Module):
    """The neural network for approximating Q-values"""

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        self.state_branch = nn.Sequential(nn.Linear(4, 32), nn.ReLU())

        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, x):
        image = x[:, :784].float() / 255.0  # Normalize pixels
        state = x[:, 784:].float()

        image_features = self.image_branch(image)
        state_features = self.state_branch(state)

        combined = torch.cat((image_features, state_features), dim=1)
        return self.combined(combined)


class ReplayMemory:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """The DQN learning agent"""

    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        self.policy_net = DQN(788, self.n_actions).to(device)
        self.target_net = DQN(788, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection"""
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1

        if evaluate or sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]], device=device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, max_steps_per_episode=70):
        rewards = []
        episode_counter = 0
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor([state], device=device)
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                total_reward += reward

                done = terminated or truncated
                next_state = torch.tensor([next_state], device=device)
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device, dtype=torch.float)

                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                self.optimize_model()

                # if episode_counter % 50 == 0:
                #     env.render()
                if done:
                    break

            episode_counter += 1
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            rewards.append(total_reward)
            print(
                f"Episode {episode + 1}: Reward = {total_reward:.2f}, Epsilon = {max(EPS_END, EPS_START * (EPS_DECAY - self.steps_done) / EPS_DECAY):.2f}"
            )

        return rewards


# Initialize environment and agent
env = gym.make(
    "RevealMNIST-v0",
    classifier_model_weights_loc="mnist_predictor_masked.pt",
    device=device,
    visualize=True,  # turn this to false when training
)

agent = DQNAgent(env)
rewards = agent.train(num_episodes=10000, max_steps_per_episode=70)
env.close()
