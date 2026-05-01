import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# Policy Network
# -----------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

# -----------------------------
# Sample action
# -----------------------------
def get_action(policy, obs):
    mean, std = policy(obs)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()
    return action.clamp(-2, 2), log_prob

# -----------------------------
# Domain randomization
# -----------------------------
def randomize_env(env):
    env.unwrapped.max_speed = np.random.uniform(6, 10)
    env.unwrapped.max_torque = np.random.uniform(1.5, 2.5)

# -----------------------------
# Sim-to-real mismatch
# -----------------------------
def modify_env(env):
    env.unwrapped.max_speed *= 1.3
    env.unwrapped.max_torque *= 0.7

# -----------------------------
# Train (REINFORCE-style)
# -----------------------------
def train(randomize=False):
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    rewards_log = []

    for episode in range(80):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)

        if randomize:
            randomize_env(env)

        log_probs = []
        rewards = []
        total_reward = 0

        for _ in range(200):
            action, log_prob = get_action(policy, obs)
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
        returns = torch.tensor(returns)

        log_probs = torch.stack(log_probs)

        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
# Run experiment
# -----------------------------
if __name__ == "__main__":
    print("Training baseline...")
    base_policy, base_rewards = train(randomize=False)

    env = gym.make("Pendulum-v1")
    print("Baseline (original):", evaluate(env, base_policy))
    print("Baseline (modified):", evaluate(env, base_policy, modify=True))

    print("\nTraining robust model...")
    robust_policy, robust_rewards = train(randomize=True)

    env = gym.make("Pendulum-v1")
    print("Robust (original):", evaluate(env, robust_policy))
    print("Robust (modified):", evaluate(env, robust_policy, modify=True))

    # Plot
    plt.plot(base_rewards, label="Baseline")
    plt.plot(robust_rewards, label="Domain Randomization")
    plt.legend()
    plt.title("Training Performance")
    plt.show()