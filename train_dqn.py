import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQNAgent 
from reveal_mnist.envs.reveal_mnist import RevealMNISTEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("RevealMNIST-v0",
               classifier_model_weights_loc="mnist_predictor_masked.pt",
               device=device,
               visualize=False)

agent = DQNAgent(state_dim=788, n_actions=env.action_space.n, device=device)

num_episodes = 900
target_update_freq = 100

env.reset(seed=42)
np.random.seed(42)
torch.manual_seed(42)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        total_reward += reward


    if episode % target_update_freq == 0:
        agent.update_target_network()

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

torch.save(agent.q_network.state_dict(), "dqn_final.pt")
print("Model saved as dqn_final.pt")