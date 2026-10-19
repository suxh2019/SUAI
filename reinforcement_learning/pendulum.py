import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
1. Train agent on Pendulum-v1
2. Modify environment (simulate real-world mismatch)
3. Show performance drop
4. Fix with domain randomization
"""

"""
What we want to show:
1) continuous control
2) RL training
3)environment mismatch
4)robustness improvement
"""
# -----------------------------
# Policy Network
# -----------------------------
class Policy(nn.Module):
    # obs_dim: dimension of the observation/state; act_dim: dimension of the action.
    def __init__(self, obs_dim, act_dim):
        # initializes the parent nn.Module class.  PyTorch needs to 
        # register the model's layers and trainable parameters.
        super().__init__()
        # Two-hidden-layer MLP
        self.net = nn.Sequential(
            # Define a neural network with two hidden layers, each containing 64 neurons.
            nn.Linear(obs_dim, 64),
            # Tanh introduces nonlinearity so that the policy can learn complex relationships between observations and actions.
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Compute the mean of the action distribution, the center of a Gaussian distribution.
        self.mean = nn.Linear(64, act_dim)
        # Create a trainable parameter representing the log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        # Process the observation
        x = self.net(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        #Return the parameters of a Gaussian policy:
        return mean, std

# -----------------------------
# Sample action
# -----------------------------
def get_action(policy, obs):
    mean, std = policy(obs)
    #mean: the center of the action distribution; creates a normal distribution for the action.
    dist = torch.distributions.Normal(mean, std) 
    # draws one random action from that distribution.
    action = dist.sample()
    # If the action has multiple dimensions, dist.log_prob(action) gives one log-probability per dimension, and .sum() combines them.
    # ecause for independent action dimensions, the joint density is the product,Taking the log turns multiplication into addition
    log_prob = dist.log_prob(action).sum()
    # clamp(-2, 2) forces the action to stay between -2 and 2.
    return action.clamp(-2, 2), log_prob

# -----------------------------
# Modify environment (sim→real)
# -----------------------------
def modify_env(env):
    # maximum speed increases by 20%.
    env.unwrapped.max_speed *= 1.2
    #maximum torque decreases by 20%
    env.unwrapped.max_torque *= 0.8

# -----------------------------
# Domain randomization
# -----------------------------
#creates a mismatch between the environment used for training and the environment used for testing.
def randomize_env(env):
    env.unwrapped.max_speed = np.random.uniform(6, 10)
    env.unwrapped.max_torque = np.random.uniform(1.5, 2.5)

# -----------------------------
# Train
# Not really PPO because it does not use PPO's probability ratio, clipping objective, advantage estimates, or old policy.
# It is closer to a basic REINFORCE / vanilla policy gradient algorithm, 
# -----------------------------
# randomize=False means normally train on the default environment.
# If randomize=True, the environment dynamics are changed by randomize_env(env)
def train(randomize=False):
    # creates the Pendulum environment. The goal is to swing the pendulum upright and keep it.
    env = gym.make("Pendulum-v1")
    # Find the dimensions.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Create my neural-network policy
    policy = Policy(obs_dim, act_dim)
    #Create an Adam optimizer. The optimizer updates the neural-network parameters after each episode.
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    # Store the total reward from each episode and later plot the learning curve.
    rewards_log = []

    for episode in range(100):
        # reset the environment.
        obs, _ = env.reset(seed=42)
        # Convert the NumPy observation into a PyTorch tensor
        obs = torch.tensor(obs, dtype=torch.float32)

        if randomize:
            randomize_env(env)
        #store information for the episode
        total_reward = 0
        log_probs = []
        rewards = []
        # run up to 200 time steps
        for _ in range(200):
            # Use the policy to sample an action
            action, log_prob = get_action(policy, obs)
            # sends the action to the environment.The environment returns: next state,reward for this action; 
            # done: episode ended normally; trunc: episode stopped due to time limit or other truncation
            # action is a PyTorch tensor.Gym expects a NumPy array. detach() removes it from PyTorch's gradient graph before converting it.
            next_obs, reward, done, trunc, _ = env.step(action.detach().numpy())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            obs = torch.tensor(next_obs, dtype=torch.float32)

            if done or trunc:
                break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        # convert the returns to a PyTorch tensor
        returns = torch.tensor(returns)
        # turn a Python list of tensors into one tensor.
        # Example: before:[tensor(-1.0), tensor(-0.8), tensor(-1.2)] after:tensor([-1.0, -0.8, -1.2])
        log_probs = torch.stack(log_probs)
        # key policy-gradient line, -\log \pi(a_t|s_t) G_t
        loss = -(log_probs * returns).mean()
        # clear old gradients.
        optimizer.zero_grad()
        #compute new gradients through backpropagation
        loss.backward()
        #update the policy parameters
        optimizer.step()
        #record the total reward for this episode
        rewards_log.append(total_reward)
        print(f"Episode {episode}, Reward {total_reward:.2f}")

    return policy, rewards_log

# -----------------------------
# Evaluate
# -----------------------------
def evaluate(env, policy, modify=False):
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        if modify:
            modify_env(env)

        total = 0
        for _ in range(200):
            action, _ = get_action(policy, obs)
            obs, r, done, trunc, _ = env.step(action.detach().numpy())
            obs = torch.tensor(obs, dtype=torch.float32)
            total += r
            if done or trunc:
                break
        rewards.append(total)

    return np.mean(rewards)

# -----------------------------
# Main function
# -----------------------------
if __name__ == "__main__":
    print("Training baseline...")
    policy_base, rewards_base = train(randomize=False)

    env = gym.make("Pendulum-v1")
    print("Baseline original:", evaluate(env, policy_base))
    print("Baseline modified:", evaluate(env, policy_base, modify=True))

    print("\nTraining robust model...")
    policy_robust, rewards_robust = train(randomize=True)

    env = gym.make("Pendulum-v1")
    print("Robust original:", evaluate(env, policy_robust))
    print("Robust modified:", evaluate(env, policy_robust, modify=True))

    # Plot
    plt.plot(rewards_base, label="Baseline")
    plt.plot(rewards_robust, label="Robust")
    plt.legend()
    plt.title("Training Rewards")
    plt.show()