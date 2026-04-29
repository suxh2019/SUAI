import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
1. Train PPO agent on Pendulum-v1
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
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
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
# Modify environment (sim→real)
# -----------------------------
def modify_env(env):
    env.unwrapped.max_speed *= 1.2
    env.unwrapped.max_torque *= 0.8

# -----------------------------
# Domain randomization
# -----------------------------
def randomize_env(env):
    env.unwrapped.max_speed = np.random.uniform(6, 10)
    env.unwrapped.max_torque = np.random.uniform(1.5, 2.5)

# -----------------------------
# Train PPO (simplified)
# -----------------------------
def train(randomize=False):
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    rewards_log = []

    for episode in range(100):
        obs, _ = env.reset(seed=42)
        obs = torch.tensor(obs, dtype=torch.float32)

        if randomize:
            randomize_env(env)

        total_reward = 0
        log_probs = []
        rewards = []

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
# Main
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