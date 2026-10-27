"""Twin Delayed DDPG (TD3) on Pendulum-v1, designed for CPU laptops.

TD3 (Twin Delayed Deep Deterministic Policy Gradient) is an off-policy, 
actor-critic reinforcement learning algorithm designed for continuous action spaces.
TD3 improves upon DDPG (Deep Deterministic Policy Gradient) by addressing one of 
its major weaknesses: overestimation of Q-values caused by function approximation errors.

TD3 introduces three important techniques:
1. Twin critics: Use two Q-networks to reduce Q-value overestimation.
2. Delayed policy updates: Update the actor less frequently than the critics.
3. Target policy smoothing: Add noise to target actions to prevent the critic from exploiting narrow, inaccurate peaks in Q-values.

Install: pip install torch numpy gymnasium
Run:     python td3_pendulum_cpu.py
"""

import copy  # Copy the online networks into independent target networks.
import random  # Sample random transitions from the replay buffer.
from collections import deque  # Store recent evaluation returns efficiently.

import numpy as np  # Handle NumPy observations, actions, and replay arrays.
import torch  # Build differentiable actor and critic networks.
import torch.nn as nn  # Define network layers and modules.
import torch.nn.functional as F  # Calculate mean squared error losses.
import torch.optim as optim  # Optimize networks with Adam.

try:
    import gymnasium as gym  # Supply the Pendulum-v1 continuous-control task.
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install torch numpy gymnasium") from exc

SEED = 42  # Fix the random seed for repeatable experiments.
ENV_NAME = "Pendulum-v1"  # Select a small continuous-action environment.
TOTAL_STEPS = 30_000  # Limit environment interaction to a laptop-friendly budget.
START_STEPS = 1_000  # Explore randomly before using the learned actor.
BATCH_SIZE = 128  # Number of transitions per gradient update.
BUFFER_CAPACITY = 100_000  # Maximum transitions kept in replay memory.
GAMMA = 0.99  # Discount factor for future rewards.
TAU = 0.005  # Polyak averaging rate for target networks.
ACTOR_LR = 3e-4  # Adam learning rate for the actor.
CRITIC_LR = 3e-4  # Adam learning rate for the two critics.
EXPL_NOISE = 0.1  # Exploration noise, expressed as a fraction of action range.
POLICY_NOISE = 0.2  # Target-policy smoothing noise, relative to action scale.
NOISE_CLIP = 0.5  # Maximum absolute target-policy smoothing noise, relative to scale.
POLICY_FREQ = 2  # Update actor and target networks every two critic updates.
EVAL_EVERY = 5_000  # Evaluate the policy after this many environment steps.
EVAL_EPISODES = 5  # Number of deterministic episodes per evaluation.
DEVICE = torch.device("cpu")  # Run exclusively on the CPU.

random.seed(SEED)  # Set Python's pseudo-random seed.
np.random.seed(SEED)  # Set NumPy's pseudo-random seed.
torch.manual_seed(SEED)  # Set PyTorch's pseudo-random seed before network creation.
torch.set_num_threads(1)  # Avoid CPU threading overhead for this tiny model.


class ReplayBuffer:  # Hold old transitions so TD3 can learn off-policy.
    def __init__(self, capacity):  # Initialize a replay buffer of bounded size.
        self.data = deque(maxlen=capacity)  # Remove oldest samples when full.

    def add(self, state, action, reward, next_state, terminated):  # Store a transition.
        self.data.append((state.copy(), action.copy(), reward, next_state.copy(), terminated))  # Copy arrays to avoid aliasing.

    def sample(self, batch_size):  # Uniformly sample a mini-batch of transitions.
        batch = random.sample(self.data, batch_size)  # Sample without replacement.
        states, actions, rewards, next_states, dones = zip(*batch)  # Separate fields.
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)  # Batch states.
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=DEVICE)  # Batch actions.
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=DEVICE).unsqueeze(1)  # Rewards shape [B, 1].
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=DEVICE)  # Batch next states.
        dones = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=DEVICE).unsqueeze(1)  # Terminal flags [B, 1].
        return states, actions, rewards, next_states, dones  # Return tensors for training.

    def __len__(self):  # Permit calls to len(replay_buffer).
        return len(self.data)  # Return the number of stored transitions.


class Actor(nn.Module):  # Learn a deterministic mapping from states to actions.
    def __init__(self, state_dim, action_dim, action_low, action_high):  # Configure sizes and limits.
        super().__init__()  # Initialize the PyTorch module.
        self.net = nn.Sequential(  # Build a small multilayer perceptron.
            nn.Linear(state_dim, 128),  # Map observations into 128 hidden features.
            nn.ReLU(),  # Add a nonlinear activation.
            nn.Linear(128, 128),  # Apply a second hidden linear layer.
            nn.ReLU(),  # Add a second nonlinear activation.
            nn.Linear(128, action_dim),  # Produce one output for each action dimension.
            nn.Tanh(),  # Constrain raw outputs to [-1, 1].
        )  # Finish building the actor network.
        low = torch.as_tensor(action_low, dtype=torch.float32)  # Convert lower action bounds to tensor.
        high = torch.as_tensor(action_high, dtype=torch.float32)  # Convert upper action bounds to tensor.
        self.register_buffer("action_scale", (high - low) / 2.0)  # Save the half-range with the model.
        self.register_buffer("action_bias", (high + low) / 2.0)  # Save the center of the action interval.

    def forward(self, state):  # Predict a bounded continuous action.
        return self.net(state) * self.action_scale + self.action_bias  # Scale tanh output to environment action bounds.


class Critic(nn.Module):  # Estimate Q(s, a) using two independent networks.
    def __init__(self, state_dim, action_dim):  # Define the dimensions of state and action.
        super().__init__()  # Initialize the module superclass.
        def make_q():  # Build one Q-network with scalar output.
            return nn.Sequential(  # Assemble fully connected layers.
                nn.Linear(state_dim + action_dim, 128),  # Combine state and action information.
                nn.ReLU(),  # Apply nonlinearity.
                nn.Linear(128, 128),  # Add a second hidden layer.
                nn.ReLU(),  # Apply nonlinearity again.
                nn.Linear(128, 1),  # Output one Q-value per transition.
            )  # End one Q-network.
        self.q1 = make_q()  # Create the first Q estimator.
        self.q2 = make_q()  # Create an independent second Q estimator.

    def forward(self, state, action):  # Calculate both Q estimates.
        sa = torch.cat([state, action], dim=1)  # Join state and action features.
        return self.q1(sa), self.q2(sa)  # Return twin Q-values.

    def q1_value(self, state, action):  # Compute only Q1 for the actor objective.
        return self.q1(torch.cat([state, action], dim=1))  # Return the first Q estimate.


class TD3:  # Implement the TD3 agent and its learning updates.
    def __init__(self, state_dim, action_dim, action_low, action_high):  # Set up agent networks.
        self.actor = Actor(state_dim, action_dim, action_low, action_high).to(DEVICE)  # Create policy network.
        self.actor_target = copy.deepcopy(self.actor)  # Create a slow-moving actor target.
        self.critic = Critic(state_dim, action_dim).to(DEVICE)  # Create two Q-networks.
        self.critic_target = copy.deepcopy(self.critic)  # Create two target Q-networks.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)  # Set actor optimizer.
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)  # Set critic optimizer.
        self.action_low = np.asarray(action_low, dtype=np.float32)  # Record minimum legal actions.
        self.action_high = np.asarray(action_high, dtype=np.float32)  # Record maximum legal actions.
        self.action_scale = (self.action_high - self.action_low) / 2.0  # Record per-dimension action scale.
        self.updates = 0  # Track the number of critic training steps.

    def select_action(self, state, noise_std=0.0):  # Choose a continuous action with optional exploration.
        with torch.no_grad():  # Disable gradient tracking during environment interaction.
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # Add batch dimension.
            action = self.actor(state_tensor).cpu().numpy()[0]  # Convert the actor's output into NumPy.
        if noise_std > 0:  # Add noise only while exploring.
            action = action + np.random.normal(0, noise_std, size=action.shape) * self.action_scale  # Perturb action.
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)  # Respect action bounds.

    def update(self, replay):  # Train critics at each update and actor less frequently.
        states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)  # Draw replay experiences.
        with torch.no_grad():  # Target computations must not build gradients.
            next_actions = self.actor_target(next_states)  # Predict next actions using target actor.
            noise = torch.randn_like(next_actions) * (POLICY_NOISE * self.actor.action_scale)  # Target smoothing noise.
            max_noise = NOISE_CLIP * self.actor.action_scale  # Compute clip bound per action component.
            noise = torch.clamp(noise, -max_noise, max_noise)  # Limit noise magnitude.
            next_actions = torch.max(torch.min(next_actions + noise, self.actor.action_bias + self.actor.action_scale), self.actor.action_bias - self.actor.action_scale)  # Clip smoothed actions.
            target_q1, target_q2 = self.critic_target(next_states, next_actions)  # Estimate twin target values.
            target_q = torch.minimum(target_q1, target_q2)  # Use lower Q-value to reduce overestimation.
            y = rewards + GAMMA * (1.0 - dones) * target_q  # Calculate Bellman targets (bootstrap at time limits).

        current_q1, current_q2 = self.critic(states, actions)  # Evaluate sampled experience using both critics.
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)  # Sum both Q-learning losses.
        self.critic_optimizer.zero_grad()  # Clear old critic gradients.
        critic_loss.backward()  # Backpropagate errors through both critics.
        self.critic_optimizer.step()  # Update both critics.
        self.updates += 1  # Record a completed critic update.

        if self.updates % POLICY_FREQ == 0:  # Delay actor and target updates relative to critics.
            actor_actions = self.actor(states)  # Compute differentiable actions at sampled states.
            actor_loss = -self.critic.q1_value(states, actor_actions).mean()  # Maximize Q1 under the actor.
            self.actor_optimizer.zero_grad()  # Clear old actor gradients.
            actor_loss.backward()  # Differentiate value through actions and actor.
            self.actor_optimizer.step()  # Update deterministic policy parameters.
            with torch.no_grad():  # Target parameters should not receive gradients.
                for online, target in zip(self.actor.parameters(), self.actor_target.parameters()):  # Visit actor weights.
                    target.mul_(1.0 - TAU).add_(TAU * online)  # Polyak-average actor target weights.
                for online, target in zip(self.critic.parameters(), self.critic_target.parameters()):  # Visit critic weights.
                    target.mul_(1.0 - TAU).add_(TAU * online)  # Polyak-average critic target weights.
        return float(critic_loss.item())  # Return critic loss for optional diagnostics.


def evaluate(agent, env, n_episodes=EVAL_EPISODES):  # Evaluate without exploration noise.
    scores = []  # Collect episode returns.
    for episode in range(n_episodes):  # Repeat evaluation over multiple episodes.
        state, _ = env.reset(seed=SEED + 1000 + episode)  # Reset with repeatable seeds.
        episode_return = 0.0  # Initialize total reward for this episode.
        while True:  # Run until environment ends the episode.
            action = agent.select_action(state)  # Choose a deterministic actor action.
            state, reward, terminated, truncated, _ = env.step(action)  # Advance the environment.
            episode_return += reward  # Accumulate received reward.
            if terminated or truncated:  # End at termination or episode time limit.
                break  # Exit the evaluation episode.
        scores.append(episode_return)  # Save one episode return.
    return float(np.mean(scores))  # Return average evaluation reward.


def train():  # Run environment interaction, training, and periodic evaluation.
    env = gym.make(ENV_NAME)  # Create the training environment.
    eval_env = gym.make(ENV_NAME)  # Use a separate environment for evaluation.
    env.action_space.seed(SEED)  # Make random exploratory actions reproducible.
    state, _ = env.reset(seed=SEED)  # Obtain the initial environment state.
    state_dim = env.observation_space.shape[0]  # Read observation feature count.
    action_dim = env.action_space.shape[0]  # Read continuous action dimensionality.
    agent = TD3(state_dim, action_dim, env.action_space.low, env.action_space.high)  # Initialize TD3.
    replay = ReplayBuffer(BUFFER_CAPACITY)  # Initialize experience storage.
    episode_return = 0.0  # Track ongoing episode reward.
    episode_number = 0  # Count completed training episodes.
    recent_returns = deque(maxlen=10)  # Keep last 10 rewards for a rolling average.
    try:  # Ensure environments close even after an interruption.
        for step in range(1, TOTAL_STEPS + 1):  # Execute the environment for a fixed budget.
            if step <= START_STEPS:  # Initially explore with random actions.
                action = env.action_space.sample()  # Sample action uniformly from legal bounds.
            else:  # Use the learned actor with Gaussian exploration noise.
                action = agent.select_action(state, noise_std=EXPL_NOISE)  # Sample a noisy deterministic action.
            next_state, reward, terminated, truncated, _ = env.step(action)  # Take one environment step.
            replay.add(state, action, reward, next_state, float(terminated))  # Treat time-limit truncations as nonterminal.
            episode_return += reward  # Add this step's reward to current return.
            state = next_state  # Move to the next observation.

            if len(replay) >= BATCH_SIZE and step > START_STEPS:  # Wait for sufficient replay data.
                agent.update(replay)  # Train the twin critics and sometimes the actor.

            if terminated or truncated:  # Reset environment at any episode boundary.
                episode_number += 1  # Increment completed episode counter.
                recent_returns.append(episode_return)  # Record the completed episode score.
                state, _ = env.reset()  # Start a new training episode.
                episode_return = 0.0  # Reset the running episode return.

            if step % EVAL_EVERY == 0:  # Periodically print deterministic evaluation results.
                eval_return = evaluate(agent, eval_env)  # Evaluate without exploration noise.
                recent_mean = np.mean(recent_returns) if recent_returns else float("nan")  # Compute recent train score.
                print(f"Step {step:6d} | Episodes {episode_number:4d} | "  # Print progress header.
                      f"Train mean (last 10): {recent_mean:8.1f} | Eval mean: {eval_return:8.1f}", flush=True)  # Print rewards.

        torch.save(agent.actor.state_dict(), "td3_pendulum_actor.pt")  # Save trained actor weights.
        print("Saved actor to td3_pendulum_actor.pt")  # Confirm model file location.
        print(f"Final evaluation return: {evaluate(agent, eval_env):.1f}")  # Print final mean reward.
    finally:  # Clean up simulation resources.
        env.close()  # Close training environment.
        eval_env.close()  # Close evaluation environment.


if __name__ == "__main__":  # Run training only when invoked as a script.
    train()  # Start TD3 training.
