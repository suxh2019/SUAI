"""CPU-friendly PPO with GAE for Gymnasium CartPole-v1.
Install: pip install torch numpy gymnasium
Run: python ppo_cartpole_cpu.py
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

SEED = 42
TOTAL_STEPS = 80_000
ROLLOUT_STEPS = 1024
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)  # small CPU networks often benefit from fewer threads


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states):
        logits = self.actor(states)
        values = self.critic(states).squeeze(-1)
        return Categorical(logits=logits), values


def train():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    obs, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    episode_return = 0.0
    recent_returns = deque(maxlen=20)
    finished_episodes = 0

    for start in range(0, TOTAL_STEPS, ROLLOUT_STEPS):
        n_steps = min(ROLLOUT_STEPS, TOTAL_STEPS - start)
        states, actions, log_probs, rewards = [], [], [], []
        values, next_values, terminateds, dones = [], [], [], []

        # Collect experience using the OLD policy (before PPO updates).
        for _ in range(n_steps):
            state_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist, value = model(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            # Bootstrap on time-limit truncation, not on genuine termination.
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(next_obs, dtype=torch.float32))

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            next_values.append(next_value)
            rewards.append(float(reward))
            terminateds.append(float(terminated))
            dones.append(float(done))

            episode_return += reward
            if done:
                recent_returns.append(episode_return)
                finished_episodes += 1
                episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        next_values = torch.stack(next_values)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminateds = torch.tensor(terminateds, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Generalized Advantage Estimation (GAE).
        # A truncation bootstraps V(next_obs), but stops GAE recursion
        # because the following saved state begins a new episode.
        advantages = torch.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            delta = (rewards[t] + GAMMA * (1.0 - terminateds[t])
                     * next_values[t] - values[t])
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # PPO multiple epochs over the same rollout.
        for _ in range(UPDATE_EPOCHS):
            order = torch.randperm(n_steps)
            for indices in order.split(MINIBATCH_SIZE):
                dist, new_values = model(states[indices])
                new_log_probs = dist.log_prob(actions[indices])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[indices]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (new_values - returns[indices]).square().mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if (start // ROLLOUT_STEPS + 1) % 5 == 0:
            avg_return = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"Steps: {start+n_steps:6d} | Episodes: {finished_episodes:4d} "
                  f"| Mean return (last 20): {avg_return:6.1f}")

    env.close()
    torch.save(model.state_dict(), "ppo_cartpole.pt")
    print("Saved trained model to ppo_cartpole.pt")
    return model


def evaluate(model, episodes=5):
    env = gym.make("CartPole-v1")
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 1000 + ep)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                dist, _ = model(torch.as_tensor(obs, dtype=torch.float32))
                action = torch.argmax(dist.probs).item()  # deterministic evaluation
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    print(f"Evaluation: mean reward {np.mean(scores):.1f} / 500 over {episodes} episodes")


if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
