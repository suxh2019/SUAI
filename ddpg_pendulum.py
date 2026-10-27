"""DDPG (Deep Deterministic Policy Gradient) for Gymnasium Pendulum-v1.

Deep Deterministic Policy Gradient (DDPG) is an off-policy, actor-critic reinforcement 
learning algorithm designed primarily for continuous action spaces.
It combines three important ideas:
- Deep Q-learning (DQN): learning a Q-function using Bellman updates and replay buffers.
- Deterministic Policy Gradient (DPG): directly optimizing a deterministic policy using gradients of the Q-function.
- Actor-Critic: training separate neural networks for selecting actions and estimating their values.

CPU-only, designed for learning and small laptop experiments.
Install: pip install torch numpy gymnasium
Run: python ddpg_pendulum_cpu.py
"""

import random  # Generate random exploration noise and sample replay transitions.
from collections import deque  # Store a fixed-size replay buffer and recent rewards.

import gymnasium as gym  # Provide the Pendulum-v1 environment.
import numpy as np  # Work with numerical arrays.
import torch  # Provide tensors and automatic differentiation.
from torch import nn  # Define the neural networks.
from torch.optim import Adam  # Optimize the actor and critic.

# ---------- Reproducibility and lightweight CPU configuration ----------
SEED = 42  # Seed for repeatable experiments.
random.seed(SEED)  # Seed Python's random generator.
np.random.seed(SEED)  # Seed NumPy's random generator.
torch.manual_seed(SEED)  # Seed PyTorch's random generator BEFORE creating models.
torch.set_num_threads(1)  # Avoid excessive CPU thread overhead on tiny networks.
DEVICE = torch.device("cpu")  # Run on a laptop CPU, even if a GPU exists.

# ---------- Hyperparameters ----------
ENV_ID = "Pendulum-v1"  # Continuous control task with 3 observations and 1 torque action.
TOTAL_STEPS = 30_000  # Total environment interactions; reduce to 5_000 for a quick test.
START_STEPS = 1_000  # Take random actions initially to populate replay memory.
BATCH_SIZE = 128  # Number of experiences per gradient update.
BUFFER_SIZE = 100_000  # Maximum transitions stored in the replay buffer.
GAMMA = 0.99  # Discount factor for future rewards.
TAU = 0.005  # Polyak averaging rate for target networks.
ACTOR_LR = 1e-3  # Learning rate for the deterministic policy.
CRITIC_LR = 1e-3  # Learning rate for the Q-value estimator.
EXPLORATION_STD = 0.1  # Gaussian exploration noise as a fraction of maximum action.
HIDDEN_SIZE = 128  # Neurons in each hidden layer.
LOG_INTERVAL = 2_000  # Frequency for training progress reports.
MODEL_PATH = "ddpg_pendulum_actor.pt"  # File used to save actor parameters.


# ---------- Replay buffer ----------
class ReplayBuffer:
    def __init__(self, capacity):  # Create a replay buffer with a maximum size.
        self.data = deque(maxlen=capacity)  # Automatically discard the oldest transition when full.

    def add(self, state, action, reward, next_state, terminated):  # Save one transition.
        self.data.append((  # Store an independent copy of the arrays.
            np.array(state, dtype=np.float32).copy(),  # Current observation.
            np.array(action, dtype=np.float32).copy(),  # Action actually taken.
            float(reward),  # Reward received.
            np.array(next_state, dtype=np.float32).copy(),  # Next observation.
            float(terminated),  # 1 for true termination; 0 for time-limit truncation.
        ))

    def sample(self, batch_size):  # Sample uncorrelated transitions for learning.
        batch = random.sample(self.data, batch_size)  # Uniform sampling without replacement.
        states, actions, rewards, next_states, terminals = zip(*batch)  # Separate columns.
        return (  # Convert all arrays into CPU tensors.
            torch.as_tensor(np.stack(states), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.stack(actions), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(terminals, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self):  # Allow len(buffer) to count transitions.
        return len(self.data)  # Return the current number of stored transitions.


# ---------- Actor: deterministic continuous-action policy ----------
class Actor(nn.Module):
    def __init__(self, state_dim, action_low, action_high):  # Initialize policy network.
        super().__init__()  # Register all PyTorch modules correctly.
        action_dim = len(action_low)  # Determine continuous action-vector size.
        self.net = nn.Sequential(  # Map states to normalized actions in [-1, 1].
            nn.Linear(state_dim, HIDDEN_SIZE),  # First fully connected layer.
            nn.ReLU(),  # Nonlinear hidden activation.
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),  # Second fully connected layer.
            nn.ReLU(),  # Nonlinear hidden activation.
            nn.Linear(HIDDEN_SIZE, action_dim),  # One output per action dimension.
            nn.Tanh(),  # Bound output to [-1, 1].
        )
        midpoint = (action_high + action_low) / 2.0  # Center of allowed action interval.
        half_range = (action_high - action_low) / 2.0  # Half-width of allowed interval.
        self.register_buffer("midpoint", torch.tensor(midpoint, dtype=torch.float32))  # Save action center.
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))  # Save scaling.

    def forward(self, states):  # Compute deterministic actions for a batch of states.
        normalized = self.net(states)  # Predict values in [-1, 1].
        return self.midpoint + self.half_range * normalized  # Scale to environment bounds.


# ---------- Critic: Q(s,a) value network ----------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):  # Initialize the Q-network.
        super().__init__()  # Initialize nn.Module.
        self.net = nn.Sequential(  # Accept concatenated observations and actions.
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),  # Input layer.
            nn.ReLU(),  # First activation.
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),  # Hidden layer.
            nn.ReLU(),  # Second activation.
            nn.Linear(HIDDEN_SIZE, 1),  # Return one estimated Q-value.
        )

    def forward(self, states, actions):  # Evaluate the given state-action pairs.
        inputs = torch.cat([states, actions], dim=-1)  # Join features and actions.
        return self.net(inputs)  # Output Q-values with shape [batch, 1].


def soft_update(target, source):  # Slowly move target weights toward online weights.
    with torch.no_grad():  # Target updates should not create a gradient graph.
        for target_param, source_param in zip(target.parameters(), source.parameters()):  # Pair layers.
            target_param.mul_(1.0 - TAU).add_(TAU * source_param)  # θ_target ← (1−τ)θ_target + τθ.


# ---------- DDPG agent ----------
class DDPGAgent:
    def __init__(self, state_dim, action_low, action_high):  # Build agent networks.
        action_dim = len(action_low)  # Number of continuous actions.
        self.actor = Actor(state_dim, action_low, action_high).to(DEVICE)  # Trainable policy.
        self.critic = Critic(state_dim, action_dim).to(DEVICE)  # Trainable Q-network.
        self.actor_target = Actor(state_dim, action_low, action_high).to(DEVICE)  # Stable target actor.
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)  # Stable target critic.
        self.actor_target.load_state_dict(self.actor.state_dict())  # Initialize targets identically.
        self.critic_target.load_state_dict(self.critic.state_dict())  # Copy initial critic weights.
        self.actor_optimizer = Adam(self.actor.parameters(), lr=ACTOR_LR)  # Actor optimizer.
        self.critic_optimizer = Adam(self.critic.parameters(), lr=CRITIC_LR)  # Critic optimizer.
        self.action_low = action_low  # Save minimum legal action values.
        self.action_high = action_high  # Save maximum legal action values.
        self.noise_scale = EXPLORATION_STD * (action_high - action_low) / 2.0  # Scale noise by action range.

    @torch.no_grad()  # Do not track gradients when choosing environment actions.
    def act(self, state, explore=True):  # Choose an action from a single observation.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # Add batch dimension.
        action = self.actor(state_tensor).squeeze(0).cpu().numpy()  # Convert to NumPy for Gymnasium.
        if explore:  # Add randomness only during training.
            action += np.random.normal(0.0, self.noise_scale, size=action.shape)  # Gaussian action noise.
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)  # Enforce action bounds.

    def update(self, replay):  # Perform one critic update and one actor update.
        states, actions, rewards, next_states, terminals = replay.sample(BATCH_SIZE)  # Sample transitions.
        with torch.no_grad():  # Keep the Bellman target fixed during this update.
            next_actions = self.actor_target(next_states)  # Get target actions at next states.
            next_q = self.critic_target(next_states, next_actions)  # Evaluate next-state actions.
            q_target = rewards + GAMMA * (1.0 - terminals) * next_q  # Bellman target.

        current_q = self.critic(states, actions)  # Estimate Q(s,a) for observed actions.
        critic_loss = nn.functional.mse_loss(current_q, q_target)  # Critic regression loss.
        self.critic_optimizer.zero_grad()  # Clear previous critic gradients.
        critic_loss.backward()  # Differentiate loss with respect to critic weights.
        self.critic_optimizer.step()  # Update critic weights.

        predicted_actions = self.actor(states)  # Produce actions from the current actor.
        actor_loss = -self.critic(states, predicted_actions).mean()  # Maximize predicted Q-values.
        self.actor_optimizer.zero_grad()  # Clear previous actor gradients.
        actor_loss.backward()  # Backpropagate through critic into the actor.
        self.actor_optimizer.step()  # Update actor weights.

        soft_update(self.actor_target, self.actor)  # Slowly update actor target network.
        soft_update(self.critic_target, self.critic)  # Slowly update critic target network.
        return critic_loss.item(), actor_loss.item()  # Report scalar losses.


def evaluate(agent, episodes=5):  # Evaluate the learned policy without exploration noise.
    env = gym.make(ENV_ID)  # Use a separate environment for evaluation.
    returns = []  # Collect per-episode rewards.
    for episode in range(episodes):  # Evaluate several independent episodes.
        state, _ = env.reset(seed=SEED + 1000 + episode)  # Reproducible starting state.
        done = False  # Track whether this episode has ended.
        episode_return = 0.0  # Reset cumulative reward.
        while not done:  # Run the current episode.
            action = agent.act(state, explore=False)  # Choose deterministic action (no noise).
            state, reward, terminated, truncated, _ = env.step(action)  # Advance environment.
            episode_return += reward  # Sum all rewards.
            done = terminated or truncated  # Stop on terminal state or time limit.
        returns.append(episode_return)  # Save this episode's reward.
    env.close()  # Release evaluation resources.
    return float(np.mean(returns))  # Return average episode reward.


def train():  # Run the complete DDPG experiment.
    env = gym.make(ENV_ID)  # Create a training environment.
    env.action_space.seed(SEED)  # Seed random actions in the action space.
    state, _ = env.reset(seed=SEED)  # Initialize training episode.
    state_dim = env.observation_space.shape[0]  # Read observation dimension.
    action_low = env.action_space.low.copy()  # Read minimum action bounds.
    action_high = env.action_space.high.copy()  # Read maximum action bounds.
    agent = DDPGAgent(state_dim, action_low, action_high)  # Initialize networks and optimizers.
    replay = ReplayBuffer(BUFFER_SIZE)  # Create experience replay storage.
    recent_rewards = deque(maxlen=20)  # Track the last 20 complete episode returns.
    episode_reward = 0.0  # Reward accumulator for the current episode.
    episodes = 0  # Number of completed training episodes.

    for step in range(1, TOTAL_STEPS + 1):  # Perform one environment interaction per iteration.
        if step <= START_STEPS:  # Initially collect broad exploratory experience.
            action = env.action_space.sample()  # Sample a uniformly random valid action.
        else:  # After warm-up, use the learned policy with added noise.
            action = agent.act(state, explore=True)  # Choose noisy deterministic action.

        next_state, reward, terminated, truncated, _ = env.step(action)  # Apply the action.
        replay.add(state, action, reward, next_state, terminated)  # Store true termination only.
        episode_reward += reward  # Accumulate the episode's undiscounted return.
        state = next_state  # Advance the current observation.

        if step > START_STEPS and len(replay) >= BATCH_SIZE:  # Train when replay is ready.
            agent.update(replay)  # Update actor, critic, and target networks.

        if terminated or truncated:  # Reset after either type of episode ending.
            recent_rewards.append(episode_reward)  # Track the completed episode's return.
            episodes += 1  # Increment episode counter.
            episode_reward = 0.0  # Reset return accumulator.
            state, _ = env.reset()  # Start a new episode.

        if step % LOG_INTERVAL == 0:  # Periodically print readable progress.
            avg = np.mean(recent_rewards) if recent_rewards else float("nan")  # Recent average.
            print(f"Steps: {step:6d} | Episodes: {episodes:4d} | "
                  f"Mean return (last 20): {avg:8.2f}")  # Training progress.

    torch.save(agent.actor.state_dict(), MODEL_PATH)  # Save trained actor parameters.
    print(f"Saved trained actor to {MODEL_PATH}")  # Confirm output location.
    print(f"Evaluation mean return (5 episodes): {evaluate(agent):.2f}")  # Evaluate policy.
    env.close()  # Close training environment.


if __name__ == "__main__":  # Only start training when running this file directly.
    train()  # Run the complete DDPG training loop.
