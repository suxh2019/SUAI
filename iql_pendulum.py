"""Offline Implicit Q-Learning (IQL) for Gymnasium Pendulum-v1, CPU only.

Install: pip install torch numpy gymnasium
Run:     python iql_pendulum_cpu.py

IQL trains ONLY from a fixed dataset. The environment is used before training to
collect that dataset, and after training to evaluate; not inside gradient updates.
This educational implementation uses twin Q critics, expectile V learning, and
advantage-weighted behavioral cloning with a tanh-Gaussian actor.
"""

import copy                              # Make independent target network copies.
import math                              # Provide pi for Gaussian log probabilities.
import random                            # Seed Python and choose behavior actions.
from collections import deque            # Track recent episode rewards compactly.

import numpy as np                       # Work with NumPy actions and observations.
import torch                             # Create tensors and compute gradients.
import torch.nn as nn                    # Define neural network building blocks.
import torch.nn.functional as F          # Use functional loss operations.
import torch.optim as optim              # Optimize neural network parameters.

try:                                     # Give an actionable message if Gymnasium is absent.
    import gymnasium as gym              # Supply the Pendulum-v1 environment.
except ImportError as exc:
    raise SystemExit("Install Gymnasium first: pip install gymnasium") from exc

# --------------------------- Experiment settings ------------------------------
SEED = 42                                # Make random initialization reproducible.
DEVICE = torch.device("cpu")             # Run without any GPU or CUDA installation.
ENV_NAME = "Pendulum-v1"                # Choose continuous-control Pendulum.
DATA_EPISODES = 120                      # Collect a modest offline training dataset.
TRAIN_STEPS = 3000                       # Number of gradient update iterations.
BATCH_SIZE = 256                         # Offline transitions in each update.
HIDDEN = 128                             # Neurons in each neural-network hidden layer.
GAMMA = 0.99                             # Discount future rewards.
TAU = 0.005                              # Target Q network Polyak update rate.
EXPECTILE = 0.7                          # IQL expectile: favor high dataset Q values.
BETA = 2.0                               # Advantage weighting strength for actor.
MAX_WEIGHT = 20.0                        # Cap exponential weights for stability.
LR = 3e-4                                # Adam learning rate for all networks.
LOG_INTERVAL = 250                       # Print losses every N gradient iterations.

random.seed(SEED)                        # Set Python's random seed.
np.random.seed(SEED)                     # Set NumPy's random seed.
torch.manual_seed(SEED)                  # Set the PyTorch random seed before model creation.
torch.set_num_threads(2)                 # Avoid excess CPU threading on small networks.

# ---------------------------- Shared neural network ---------------------------
class MLP(nn.Module):                     # Define a reusable fully connected network.
    def __init__(self, input_dim, output_dim):  # Specify input and output dimensions.
        super().__init__()               # Initialize the nn.Module base class.
        self.net = nn.Sequential(         # Run layers from top to bottom.
            nn.Linear(input_dim, HIDDEN), # First fully connected layer.
            nn.ReLU(),                   # Introduce a nonlinear activation.
            nn.Linear(HIDDEN, HIDDEN),    # Second fully connected layer.
            nn.ReLU(),                   # Introduce another nonlinear activation.
            nn.Linear(HIDDEN, output_dim) # Produce final values or actor parameters.
        )

    def forward(self, x):                 # Define how inputs pass through the network.
        return self.net(x)               # Return the network output.


class Actor(nn.Module):                   # Stochastic Gaussian policy for continuous actions.
    def __init__(self, obs_dim, act_dim, action_limit):
        super().__init__()               # Register PyTorch model components.
        self.network = MLP(obs_dim, 2 * act_dim)  # Predict Gaussian mean and log std.
        self.action_limit = action_limit # Convert tanh outputs to valid action range.

    def distribution(self, states):       # Build the Gaussian before tanh squashing.
        mean, log_std = self.network(states).chunk(2, dim=-1)  # Split parameters.
        log_std = torch.clamp(log_std, -5.0, 2.0) # Prevent extreme standard deviations.
        std = log_std.exp()               # Convert logarithmic std to positive std.
        return torch.distributions.Normal(mean, std) # Independent Normal per dimension.

    def act(self, state, deterministic=True): # Choose an environment action.
        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():             # Avoid tracking derivatives for environment use.
            dist = self.distribution(x)   # Obtain the current action distribution.
            raw_action = dist.mean if deterministic else dist.sample() # Mean or random.
            bounded = torch.tanh(raw_action) * self.action_limit # Restrict torque.
        return bounded.squeeze(0).cpu().numpy() # Return NumPy action for Gymnasium.

    def log_prob(self, states, actions):   # Compute log policy density of DATASET actions.
        dist = self.distribution(states)  # Calculate Normal parameters for each state.
        scaled = actions / self.action_limit # Normalize actions into approximately [-1,1].
        scaled = torch.clamp(scaled, -0.999999, 0.999999) # Avoid infinite atanh.
        raw = torch.atanh(scaled)         # Undo tanh to evaluate Gaussian density.
        gaussian_logp = dist.log_prob(raw) # Compute per-dimension Gaussian log density.
        correction = torch.log(          # Account for tanh and action scaling Jacobian.
            self.action_limit * (1.0 - scaled.square()) + 1e-6
        )
        return (gaussian_logp - correction).sum(dim=-1, keepdim=True) # [batch, 1].


class Critic(nn.Module):                  # Estimate Q(s,a), expected discounted return.
    def __init__(self, obs_dim, act_dim):
        super().__init__()               # Initialize the PyTorch module.
        self.network = MLP(obs_dim + act_dim, 1) # Output one scalar Q value.

    def forward(self, states, actions):   # Receive state and action tensors.
        return self.network(torch.cat((states, actions), dim=-1)) # Concatenate features.


# ----------------------- Offline dataset collection ---------------------------
def behavior_action(observation, action_limit): # A simple, imperfect behavior policy.
    cos_theta, sin_theta, angular_speed = observation # Read Pendulum-v1 state features.
    theta = np.arctan2(sin_theta, cos_theta) # Recover angle from sin and cos.
    if np.random.rand() < 0.30:          # Mix in some random actions for coverage.
        return np.array([np.random.uniform(-action_limit, action_limit)], dtype=np.float32)
    torque = -2.0 * theta - 0.7 * angular_speed # Simple PD-like control action.
    torque += np.random.normal(0, 0.65) # Add exploration noise to the behavior policy.
    torque = np.clip(torque, -action_limit, action_limit) # Respect action bounds.
    return np.array([torque], dtype=np.float32) # Gym expects a 1D action array.


def collect_offline_data(env, action_limit): # Fill replay data ONCE before training.
    states, actions, rewards = [], [], [] # Separate lists for collected values.
    next_states, terminals = [], []     # Keep successor observations and termination flags.
    episode_returns = []                # Track quality of collected behavior trajectories.
    for episode in range(DATA_EPISODES): # Repeat for the desired dataset size.
        state, _ = env.reset(seed=SEED + episode) # Initialize an episode reproducibly.
        total_reward = 0.0              # Reset the episode return accumulator.
        while True:                      # Continue until an episode ends.
            action = behavior_action(state, action_limit) # Sample a behavior action.
            next_state, reward, terminated, truncated, _ = env.step(action) # Apply action.
            states.append(state.copy())  # Save the current observation.
            actions.append(action.copy()) # Save the chosen continuous action.
            rewards.append([reward])     # Keep rewards in [N,1] tensor shape.
            next_states.append(next_state.copy()) # Save the successor observation.
            terminals.append([float(terminated)]) # Bootstrap on time-limit truncation.
            total_reward += reward       # Accumulate episode reward for diagnostics.
            state = next_state           # Move on to the following observation.
            if terminated or truncated:  # Finish episodes on either Gymnasium signal.
                break                    # Start collection of the next episode.
        episode_returns.append(total_reward) # Remember the behavior episode return.

    def as_tensor(data):                  # Convert collected values to CPU tensors.
        return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=DEVICE)

    dataset = (                            # Store tensors in convenient fixed order.
        as_tensor(states),                # Shape [N, observation_dim].
        as_tensor(actions),               # Shape [N, action_dim].
        as_tensor(rewards),               # Shape [N, 1].
        as_tensor(next_states),           # Shape [N, observation_dim].
        as_tensor(terminals),             # Shape [N, 1].
    )
    return dataset, episode_returns       # IQL will only sample from this fixed dataset.


# ------------------------- Implicit Q-Learning agent --------------------------
class IQL:
    def __init__(self, obs_dim, act_dim, action_limit):
        self.actor = Actor(obs_dim, act_dim, action_limit).to(DEVICE) # Policy pi(a|s).
        self.q1 = Critic(obs_dim, act_dim).to(DEVICE) # First Q function.
        self.q2 = Critic(obs_dim, act_dim).to(DEVICE) # Second Q function.
        self.value = MLP(obs_dim, 1).to(DEVICE) # Expectile state value V(s).
        self.q1_target = copy.deepcopy(self.q1) # Slowly updated Q1 target.
        self.q2_target = copy.deepcopy(self.q2) # Slowly updated Q2 target.
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR) # Actor optimizer.
        self.q_opt = optim.Adam(          # Joint optimizer for twin Q networks.
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR
        )
        self.value_opt = optim.Adam(self.value.parameters(), lr=LR) # Value optimizer.

    @staticmethod
    def soft_update(online, target):      # Polyak-average target network parameters.
        with torch.no_grad():             # Target weights should not get gradients.
            for p, p_target in zip(online.parameters(), target.parameters()):
                p_target.mul_(1.0 - TAU).add_(TAU * p) # target <- (1-tau)target+tau online.

    def update(self, batch):              # Perform one IQL update from offline data.
        s, a, r, next_s, done = batch     # Unpack state, action, reward, next state, done.

        # 1. Fit expectile V(s) to the dataset-action twin-Q estimate.
        with torch.no_grad():             # Treat the Q estimate as a fixed target.
            q_dataset = torch.minimum(self.q1_target(s, a), self.q2_target(s, a))
        v = self.value(s)                 # Predict V(s) for the sampled states.
        diff = q_dataset - v             # Positive difference means Q exceeds V.
        weights = torch.where(diff > 0, EXPECTILE, 1.0 - EXPECTILE) # Asymmetric weighting.
        value_loss = (weights * diff.square()).mean() # Expectile regression loss.
        self.value_opt.zero_grad()       # Clear existing value-network gradients.
        value_loss.backward()            # Differentiate the value regression loss.
        self.value_opt.step()            # Update only the value network.

        # 2. Train both critics with TD targets using V(next_state), NOT sampled actions.
        with torch.no_grad():             # Do not differentiate through the TD target.
            td_target = r + GAMMA * (1.0 - done) * self.value(next_s)
        q1_pred = self.q1(s, a)          # Predict first critic's Q(s,a).
        q2_pred = self.q2(s, a)          # Predict second critic's Q(s,a).
        q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        self.q_opt.zero_grad()           # Clear stale critic gradients.
        q_loss.backward()                # Differentiate both squared TD errors.
        self.q_opt.step()                # Update both Q networks.

        # 3. Fit actor by advantage-weighted behavioral cloning on DATASET actions.
        with torch.no_grad():             # Advantages are fixed weights for this update.
            q_min = torch.minimum(self.q1_target(s, a), self.q2_target(s, a))
            advantage = q_min - self.value(s) # A(s,a) = Q(s,a) - V(s).
            exp_weight = torch.exp(BETA * advantage).clamp(max=MAX_WEIGHT)
        logp = self.actor.log_prob(s, a) # Log probability of observed dataset actions.
        actor_loss = -(exp_weight * logp).mean() # Weighted behavior cloning objective.
        self.actor_opt.zero_grad()       # Reset policy gradients.
        actor_loss.backward()            # Differentiate weighted log-likelihood.
        self.actor_opt.step()            # Improve probability of high-advantage actions.

        self.soft_update(self.q1, self.q1_target) # Slowly synchronize target Q1.
        self.soft_update(self.q2, self.q2_target) # Slowly synchronize target Q2.
        return (value_loss.item(), q_loss.item(), actor_loss.item()) # Training diagnostics.


def evaluate(env, actor, episodes=5):   # Evaluate the learned policy without training.
    returns = []                         # Store a return for each evaluation episode.
    for episode in range(episodes):      # Repeat to reduce evaluation noise.
        state, _ = env.reset(seed=10000 + episode) # Use separate evaluation seeds.
        total = 0.0                      # Start a new return accumulator.
        while True:                      # Advance the environment for one episode.
            action = actor.act(state, deterministic=True) # Use mean policy action.
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward              # Accumulate reward.
            if terminated or truncated:  # Stop at true termination or time limit.
                break
        returns.append(total)            # Record this episode's return.
    return float(np.mean(returns))       # Return average reward across episodes.


def main():                             # Entry point for collecting, training, testing.
    env = gym.make(ENV_NAME)              # Make environment for offline data collection.
    eval_env = gym.make(ENV_NAME)         # Use a separate environment for evaluation.
    obs_dim = env.observation_space.shape[0] # Pendulum has three observation features.
    act_dim = env.action_space.shape[0] # Pendulum has one continuous action.
    assert np.allclose(env.action_space.high, -env.action_space.low) # Symmetric range.
    assert np.allclose(env.action_space.high, env.action_space.high[0]) # Shared action limit.
    action_limit = float(env.action_space.high[0]) # For Pendulum, torque magnitude is 2.

    print("Collecting a fixed offline dataset...") # Mark data-collection phase.
    dataset, returns = collect_offline_data(env, action_limit) # No model updates here.
    print(f"Transitions: {len(dataset[0]):,}; behavior mean return: {np.mean(returns):.1f}")
    env.close()                          # No further data collection during training.

    agent = IQL(obs_dim, act_dim, action_limit) # Construct neural networks and optimizers.
    initial_return = evaluate(eval_env, agent.actor) # Evaluate untrained actor.
    print(f"Initial policy evaluation: {initial_return:.1f}")
    print("Training IQL from the fixed dataset (no environment steps)...")
    N = len(dataset[0])                  # Count stored offline transitions.
    for step in range(1, TRAIN_STEPS + 1): # Perform fixed number of gradient updates.
        indices = torch.randint(0, N, (BATCH_SIZE,), device=DEVICE) # Sample with replacement.
        batch = tuple(field[indices] for field in dataset) # Build tensor mini-batch.
        value_loss, q_loss, actor_loss = agent.update(batch) # Execute one IQL update.
        if step % LOG_INTERVAL == 0 or step == 1: # Print occasional metrics.
            print(f"Update {step:4d}/{TRAIN_STEPS} | V loss {value_loss:.4f} | "
                  f"Q loss {q_loss:.4f} | actor loss {actor_loss:.4f}")

    final_return = evaluate(eval_env, agent.actor) # Evaluate policy after training.
    print(f"Final mean evaluation return: {final_return:.1f}") # Higher is better (max 0).
    torch.save(agent.actor.state_dict(), "iql_pendulum_actor.pt") # Save trained weights.
    print("Saved actor parameters to iql_pendulum_actor.pt") # Confirm local output.
    eval_env.close()                     # Release the evaluation environment.


if __name__ == "__main__":              # Only train when executed as a script.
    main()                                # Launch the end-to-end experiment.
