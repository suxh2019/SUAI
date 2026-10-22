"""CPU-only continuous-action Conservative Q-Learning (CQL) example.

Install: pip install torch numpy gymnasium
Run:     python cql_pendulum_cpu.py

This uses a fixed offline dataset collected by a heuristic/noisy controller.
CQL uses a conservative penalty on sampled actions plus SAC-style actor updates.
The simple dataset is educational; high returns are not guaranteed.
"""
import copy                          # Copy networks when initializing target critics.
import random                        # Seed Python's random generator.
from collections import deque        # Store recent evaluation information if desired.

import gymnasium as gym              # Pendulum-v1 continuous-control environment.
import numpy as np                   # Dataset storage and random sampling.
import torch                         # Neural networks and automatic differentiation.
import torch.nn as nn                # Neural network building blocks.
import torch.nn.functional as F      # Losses and activation functions.
from torch.distributions import Normal  # Gaussian actor distribution.

SEED = 42                           # Reproducible random initialization and sampling.
DEVICE = torch.device('cpu')         # Use laptop CPU; no GPU required.
DATA_EPISODES = 120                  # Number of trajectories in the fixed offline dataset.
TRAIN_STEPS = 3000                   # Number of offline gradient updates.
BATCH_SIZE = 128                     # Transitions sampled for each update.
HIDDEN_SIZE = 128                    # Width of small fully connected networks.
GAMMA = 0.99                        # Discount future rewards.
TAU = 0.005                         # Target critic Polyak update factor.
LR = 3e-4                           # Learning rate for actor, critics, and alpha.
CQL_WEIGHT = 1.0                    # Strength of the conservative Q penalty.
CQL_N_ACTIONS = 10                  # Random/policy actions sampled per state for CQL.
LOG_STD_MIN = -5.0                  # Lower bound for Gaussian log standard deviation.
LOG_STD_MAX = 2.0                   # Upper bound for Gaussian log standard deviation.
EVAL_EVERY = 500                    # Evaluate after this many gradient updates.

random.seed(SEED)                   # Seed Python randomness.
np.random.seed(SEED)                # Seed NumPy randomness.
torch.manual_seed(SEED)              # Seed PyTorch randomness before network creation.
torch.set_num_threads(1)             # Reduce overhead for small CPU neural networks.


def mlp(in_dim, out_dim):           # Construct a compact fully connected network.
    return nn.Sequential(           # Apply layers in sequence.
        nn.Linear(in_dim, HIDDEN_SIZE),  # First linear layer.
        nn.ReLU(),                   # First nonlinear activation.
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),  # Second linear layer.
        nn.ReLU(),                   # Second nonlinear activation.
        nn.Linear(HIDDEN_SIZE, out_dim),  # Output layer.
    )


class GaussianActor(nn.Module):      # Stochastic actor used by SAC-style CQL.
    def __init__(self, obs_dim, act_dim, action_low, action_high):
        super().__init__()           # Register PyTorch submodules.
        self.body = nn.Sequential(   # Map states to learned features.
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_SIZE, act_dim)  # Gaussian location.
        self.log_std = nn.Linear(HIDDEN_SIZE, act_dim)  # Gaussian spread.
        low = torch.as_tensor(action_low, dtype=torch.float32)  # Lower action bounds.
        high = torch.as_tensor(action_high, dtype=torch.float32)  # Upper action bounds.
        self.register_buffer('scale', (high - low) / 2)  # Scale tanh output.
        self.register_buffer('bias', (high + low) / 2)  # Shift tanh output.

    def forward(self, obs, deterministic=False, with_logprob=True):
        features = self.body(obs)    # Encode the observations.
        mu = self.mean(features)     # Gaussian mean before tanh.
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()          # Convert log standard deviation into std.
        distribution = Normal(mu, std)  # Create differentiable Gaussian distribution.
        z = mu if deterministic else distribution.rsample()  # Reparameterized sampling.
        squashed = torch.tanh(z)     # Bound actions to [-1, 1].
        action = squashed * self.scale + self.bias  # Scale to environment limits.
        log_prob = None              # Deterministic evaluation does not need log density.
        if with_logprob:             # Compute log probability with tanh correction.
            log_prob = distribution.log_prob(z)  # Unsquashed Gaussian log density.
            correction = 2 * (np.log(2) - z - F.softplus(-2 * z))
            log_prob = (log_prob - correction - torch.log(self.scale)).sum(-1, keepdim=True)
        return action, log_prob      # Actions [batch, act_dim], logp [batch, 1].


class Critic(nn.Module):             # Q(s,a) maps state-action pairs to expected value.
    def __init__(self, obs_dim, act_dim):
        super().__init__()           # Initialize nn.Module.
        self.network = mlp(obs_dim + act_dim, 1)  # One Q-value per state-action pair.

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))  # Join observation and action.


def collect_offline_dataset(env):   # Interact only BEFORE offline training starts.
    transitions = []                # Hold tuples of (s,a,r,s',terminal).
    for episode in range(DATA_EPISODES):  # Collect a fixed number of episodes.
        obs, _ = env.reset(seed=SEED + episode)  # Reset with different episode seeds.
        finished = False            # Track episode completion.
        while not finished:         # Gather one complete trajectory.
            theta = np.arctan2(obs[1], obs[0])  # Recover pendulum angle from cos/sin.
            angular_velocity = obs[2]  # Obtain pendulum angular velocity.
            # Heuristic feedback plus exploration creates more useful data than pure random.
            torque = -2.0 * theta - 0.6 * angular_velocity
            torque += np.random.normal(0.0, 0.8)  # Add behavior-policy exploration.
            action = np.clip(np.array([torque], dtype=np.float32),
                             env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # True termination stops value bootstrapping; time limits are truncations.
            transitions.append((obs.copy(), action.copy(), reward,
                                next_obs.copy(), float(terminated)))
            obs = next_obs           # Move to the following state.
            finished = terminated or truncated  # Stop at either episode-ending signal.
    states, actions, rewards, next_states, terminals = zip(*transitions)
    return (np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(terminals, dtype=np.float32).reshape(-1, 1))


def sample_batch(dataset):         # Draw an independent minibatch from fixed data.
    indices = np.random.randint(0, len(dataset[0]), size=BATCH_SIZE)
    return [torch.as_tensor(array[indices], device=DEVICE) for array in dataset]


class CQLAgent:                      # Keep all trainable networks and optimizers together.
    def __init__(self, obs_dim, act_dim, low, high):
        self.actor = GaussianActor(obs_dim, act_dim, low, high).to(DEVICE)
        self.q1 = Critic(obs_dim, act_dim).to(DEVICE)  # First critic.
        self.q2 = Critic(obs_dim, act_dim).to(DEVICE)  # Second critic.
        self.q1_target = copy.deepcopy(self.q1)  # Slowly updated first target critic.
        self.q2_target = copy.deepcopy(self.q2)  # Slowly updated second target critic.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -float(act_dim)  # Conventional continuous SAC target entropy.
        self.act_dim = act_dim      # Number of action coordinates.
        self.low = torch.as_tensor(low, dtype=torch.float32, device=DEVICE)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=DEVICE)

    def act(self, obs):              # Select deterministic action for evaluation.
        with torch.no_grad():        # Do not construct gradient graphs when evaluating.
            observation = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _ = self.actor(observation, deterministic=True, with_logprob=False)
        return action.squeeze(0).cpu().numpy()  # Return NumPy action to Gymnasium.

    def conservative_loss(self, critic, states, next_states, dataset_actions, alpha):
        batch = states.shape[0]      # Number of offline transitions in this update.
        n = CQL_N_ACTIONS           # Number of candidate actions per state.
        expanded_states = states.unsqueeze(1).expand(-1, n, -1)
        expanded_next = next_states.unsqueeze(1).expand(-1, n, -1)
        flat_states = expanded_states.reshape(batch * n, -1)
        flat_next = expanded_next.reshape(batch * n, -1)
        # Uniform candidate actions explore the entire bounded action domain.
        random_actions = self.low + torch.rand(
            (batch * n, self.act_dim), device=DEVICE
        ) * (self.high - self.low)
        # Draw policy actions without allowing the CQL critic loss to update the actor.
        with torch.no_grad():
            policy_actions, policy_logp = self.actor(flat_states)
            next_actions, next_logp = self.actor(flat_next)
        q_uniform = critic(flat_states, random_actions).reshape(batch, n, 1)
        q_policy = critic(flat_states, policy_actions).reshape(batch, n, 1)
        q_next = critic(flat_next, next_actions).reshape(batch, n, 1)
        # Continuous CQL uses importance-sampling log densities inside logsumexp.
        uniform_log_density = -torch.log(self.high - self.low).sum()
        cat_q = torch.cat([
            q_uniform - uniform_log_density,
            q_policy - policy_logp.reshape(batch, n, 1),
            q_next - next_logp.reshape(batch, n, 1),
        ], dim=1)
        # logmeanexp provides a conservative estimate over candidate Q-values.
        conservative_term = torch.logsumexp(cat_q, dim=1).mean() - np.log(3 * n)
        # Subtract the dataset's estimated Q-value to preserve in-distribution actions.
        data_term = critic(states, dataset_actions).mean()
        return CQL_WEIGHT * (conservative_term - data_term)

    def update(self, batch):         # One offline gradient update of critics and actor.
        states, actions, rewards, next_states, terminals = batch
        alpha = self.log_alpha.exp().detach()  # Treat alpha as constant for actor/critic.
        with torch.no_grad():        # Bellman target must not backprop through target networks.
            next_actions, next_logp = self.actor(next_states)
            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_value = torch.minimum(next_q1, next_q2) - alpha * next_logp
            target_q = rewards + GAMMA * (1.0 - terminals) * next_value
        q1_pred = self.q1(states, actions)  # First Q estimate on dataset transitions.
        q2_pred = self.q2(states, actions)  # Second Q estimate on dataset transitions.
        bellman_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        cql1 = self.conservative_loss(self.q1, states, next_states, actions, alpha)
        cql2 = self.conservative_loss(self.q2, states, next_states, actions, alpha)
        critic_loss = bellman_loss + cql1 + cql2  # Conservative Bellman optimization.
        self.q_opt.zero_grad()       # Clear old critic gradients.
        critic_loss.backward()       # Compute critic gradients.
        self.q_opt.step()           # Update both critics.
        sampled_actions, logp = self.actor(states)  # Reparameterized policy actions.
        q_pi = torch.minimum(self.q1(states, sampled_actions),
                             self.q2(states, sampled_actions))
        actor_loss = (alpha * logp - q_pi).mean()  # SAC maximum-entropy actor loss.
        self.actor_opt.zero_grad()   # Clear prior actor gradients.
        actor_loss.backward()        # Differentiate actor through its sampled actions.
        self.actor_opt.step()        # Improve the actor.
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()   # Clear old temperature gradients.
        alpha_loss.backward()        # Calculate temperature gradient.
        self.alpha_opt.step()        # Tune entropy coefficient.
        with torch.no_grad():        # Softly synchronize target critics after each update.
            for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                for p, target_p in zip(current.parameters(), target.parameters()):
                    target_p.mul_(1 - TAU).add_(TAU * p)
        return float(critic_loss.item()), float(actor_loss.item()), float(alpha.item())


def evaluate(agent, env, n_episodes=5):  # Test the policy without training it.
    scores = []                      # Collect total rewards for each episode.
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + episode)  # Repeatable evaluation states.
        done = False                 # Track whether evaluation episode ended.
        total = 0.0                 # Accumulate episode reward.
        while not done:             # Continue until termination or time limit.
            action = agent.act(obs)  # Choose deterministic mean action.
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward         # Add reward for this transition.
            done = terminated or truncated
        scores.append(total)         # Record completed return.
    return float(np.mean(scores))    # Average evaluation return.


def main():                         # Execute the full offline RL experiment.
    env = gym.make('Pendulum-v1')     # Build data-collection environment.
    eval_env = gym.make('Pendulum-v1')  # Separate evaluation environment.
    env.action_space.seed(SEED)      # Make action-space draws reproducible.
    obs_dim = env.observation_space.shape[0]  # Pendulum has 3 state features.
    act_dim = env.action_space.shape[0]  # Pendulum has 1 continuous action.
    print('Collecting fixed offline dataset...')
    dataset = collect_offline_dataset(env)  # No more environment steps for training.
    print(f'Collected {len(dataset[0]):,} transitions.')
    agent = CQLAgent(obs_dim, act_dim, env.action_space.low, env.action_space.high)
    for step in range(1, TRAIN_STEPS + 1):  # Iterate over offline gradient updates.
        batch = sample_batch(dataset)       # Only draw experience from fixed dataset.
        critic_loss, actor_loss, alpha = agent.update(batch)
        if step % EVAL_EVERY == 0 or step == 1:
            avg_return = evaluate(agent, eval_env)  # Evaluate but do not add transitions.
            print(f'Step {step:5d} | critic loss {critic_loss:8.3f} | '
                  f'actor loss {actor_loss:8.3f} | alpha {alpha:.3f} | '
                  f'eval return {avg_return:8.1f}')
    torch.save(agent.actor.state_dict(), 'cql_pendulum_actor.pt')  # Save learned actor.
    print('Saved actor to cql_pendulum_actor.pt')
    env.close()                       # Release environment resources.
    eval_env.close()                  # Release evaluation environment resources.


if __name__ == '__main__':            # Run main only when executed as a script.
    main()                             # Start data collection and offline learning.
