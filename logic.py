'''
06.05.2025
created first by yigit
as an inital draft
'''
import torch
import numpy as np
from dqn_agent import DQNAgent
from reveal_mnist.envs.reveal_mnist import RevealMNISTEnv
import gymnasium as gym 

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#removed different init. of env, will continue with under~ env.
env = gym.make('RevealMNIST-v0',
               classifier_model_weights_loc="mnist_predictor_masked.pt",
               device=device,
               visualize=True)

# Agent settings
# Removed the unused settings(n_actions).
n_episodes = 1000
#  the final grid on which the agent traverses in 7x7. it "can" reveal 49 patches, can't it?
max_steps_per_episode = 49  # At most 49 patches can be revealed

# Basic action selection: always choose the action with highest predicted class score

# This function is not uselful anymore, will continue with select_action_alt.
# def select_action(observation):
#     image_data = observation[:784]
#     image_tensor = torch.tensor(
#         image_data, dtype=torch.float32).view(1, 1, 28, 28).to(device)

#     with torch.no_grad():
#         logits = MNISTPredictor(image_tensor)  # Output logits for digit classes
#         # Select the most likely digit class
#         action = torch.argmax(logits).item()

#     return action


def select_action_alt(obs):
    '''
    selects the action according to the policy.
    assuming a greedy approach, it should take the aciton for that state with the highest value
    take the value of each action for that state (=obs)
    return the action with the highest value
    '''

    action_values = {}
    action_space = range(40, 49) # To test some results
    for action in action_space:
        action_values[action] = action_value_for_state(obs, action)

    # I cannot code "pythonic"
    _, greedy_action = max(action_values.items(), key=lambda item: item[1])

    return greedy_action


def action_value_for_state(state, action) -> float:
    '''
    returns the value for a state-action pair 
    uses the DQN parameters (for such an implementation)
    '''
    # TODO: implement (somehow...)
    return 0


# Run the baseline agent
for episode in range(n_episodes):
    obs, info = env.reset()
    total_reward: float = 0

    for step in range(max_steps_per_episode):

        action = select_action_alt(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated:
            break

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
