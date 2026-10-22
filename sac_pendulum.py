"""CPU-friendly Soft Actor-Critic (SAC) for Gymnasium Pendulum-v1.

Install: pip install torch numpy gymnasium
Run:     python sac_pendulum_cpu.py

SAC: off-policy replay, twin Q critics, target critics, squashed Gaussian
actor, entropy regularization with automatically learned temperature.
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

SEED = 42
TOTAL_STEPS = 30_000        # Try 5_000 for a quick smoke test
START_STEPS = 1_000        # Random exploration
UPDATE_AFTER = 1_000
BATCH_SIZE = 128
REPLAY_SIZE = 100_000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
HIDDEN_SIZE = 128
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EVAL_INTERVAL = 5_000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)  # Usually helps small CPU networks; tune if needed
DEVICE = torch.device('cpu')


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, terminated):
        self.data.append((np.array(obs, copy=True), np.array(action, copy=True),
                          reward, np.array(next_obs, copy=True), float(terminated)))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        obs, action, reward, next_obs, terminated = zip(*batch)
        return (torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(np.asarray(action), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
                torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(terminated, dtype=torch.float32, device=DEVICE).unsqueeze(-1))

    def __len__(self):
        return len(self.data)


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, HIDDEN_SIZE), nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, output_dim))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.register_buffer('action_scale', torch.as_tensor((act_high - act_low) / 2, dtype=torch.float32))
        self.register_buffer('action_bias', torch.as_tensor((act_high + act_low) / 2, dtype=torch.float32))

    def forward(self, obs, deterministic=False, with_logprob=True):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else dist.rsample()  # Reparameterization trick
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias
        log_prob = None
        if with_logprob:
            # Change-of-variables correction for tanh squashing and action scaling.
            correction = 2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))
            log_prob = (dist.log_prob(raw_action) - correction -
                        torch.log(self.action_scale)).sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


@torch.no_grad()
def soft_update(source, target):
    for src, tgt in zip(source.parameters(), target.parameters()):
        tgt.mul_(1 - TAU).add_(TAU * src)


@torch.no_grad()
def evaluate(actor, episodes=3):
    env = gym.make('Pendulum-v1')
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=SEED + 1_000 + i)
        score = 0.0
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = actor(x, deterministic=True, with_logprob=False)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            score += reward
            done = terminated or truncated
        scores.append(score)
    env.close()
    return float(np.mean(scores))


def main():
    env = gym.make('Pendulum-v1')
    env.action_space.seed(SEED)
    obs, _ = env.reset(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim, env.action_space.low, env.action_space.high).to(DEVICE)
    q1 = Critic(obs_dim, act_dim).to(DEVICE)
    q2 = Critic(obs_dim, act_dim).to(DEVICE)
    q1_target = Critic(obs_dim, act_dim).to(DEVICE)
    q2_target = Critic(obs_dim, act_dim).to(DEVICE)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for target in (q1_target, q2_target):
        target.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
    log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = -float(act_dim)
    replay = ReplayBuffer(REPLAY_SIZE)
    episode_return, episode_length = 0.0, 0

    for step in range(1, TOTAL_STEPS + 1):
        if step <= START_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action, _ = actor(x, with_logprob=False)
                action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated)
        episode_return += reward
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            print(f'Step {step:6d} | Episode reward {episode_return:8.1f} | Length {episode_length}')
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0

        if step >= UPDATE_AFTER and len(replay) >= BATCH_SIZE:
            s, a, r, s_next, terminated_batch = replay.sample(BATCH_SIZE)

            # Critic targets: bootstrapping stops on true terminations, not time limits.
            with torch.no_grad():
                next_a, next_logp = actor(s_next)
                target_q = torch.min(q1_target(s_next, next_a), q2_target(s_next, next_a))
                target_q = r + GAMMA * (1 - terminated_batch) * (target_q - log_alpha.exp() * next_logp)

            critic_loss = F.mse_loss(q1(s, a), target_q) + F.mse_loss(q2(s, a), target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Temporarily freeze critics; gradients still flow through actions to actor.
            for critic in (q1, q2):
                critic.requires_grad_(False)
            new_a, logp = actor(s)
            min_q = torch.min(q1(s, new_a), q2(s, new_a))
            actor_loss = (log_alpha.exp().detach() * logp - min_q).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            for critic in (q1, q2):
                critic.requires_grad_(True)

            alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            soft_update(q1, q1_target)
            soft_update(q2, q2_target)

        if step % EVAL_INTERVAL == 0:
            score = evaluate(actor)
            print(f'>>> Step {step} | Eval reward {score:.1f} | alpha {log_alpha.exp().item():.3f}')

    torch.save(actor.state_dict(), 'sac_pendulum_actor.pt')
    env.close()
    print('Saved actor weights to sac_pendulum_actor.pt')


if __name__ == '__main__':
    main()
