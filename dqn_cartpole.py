"""CPU-friendly Deep Q-Network (DQN) for Gymnasium CartPole-v1.
Install: pip install torch numpy gymnasium
Run: python dqn_cartpole_cpu.py
"""

import random  # Provide random sampling for exploration and experience replay.
from collections import deque  # Store a fixed-size queue of recent transitions.

import gymnasium as gym  # Provide the CartPole reinforcement learning environment.
import numpy as np  # Support numerical arrays and random sampling.
import torch  # Provide tensors and automatic differentiation.
import torch.nn as nn  # Provide neural network layers and loss functions.
import torch.optim as optim  # Provide the Adam optimizer.

SEED = 42  # Make experiments repeatable under comparable conditions.
EPISODES = 350  # Maximum number of training episodes.
MAX_STEPS = 500  # Maximum steps per CartPole episode.
GAMMA = 0.99  # Discount factor for future rewards.
LEARNING_RATE = 1e-3  # Adam learning rate.
BATCH_SIZE = 64  # Number of transitions sampled for one update.
BUFFER_SIZE = 20_000  # Maximum number of saved transitions.
MIN_REPLAY_SIZE = 1_000  # Wait for this many transitions before training.
TARGET_UPDATE_INTERVAL = 250  # Copy online weights to target every 250 optimizer steps.
EPSILON_START = 1.0  # Start with fully random actions.
EPSILON_END = 0.05  # Keep a small amount of exploration.
EPSILON_DECAY_STEPS = 20_000  # Gradually reduce epsilon over environment steps.
DEVICE = torch.device('cpu')  # Force training to run on a regular laptop CPU.

random.seed(SEED)  # Seed Python's random generator.
np.random.seed(SEED)  # Seed NumPy's random generator.
torch.manual_seed(SEED)  # Seed PyTorch's random generator before creating networks.
torch.set_num_threads(2)  # Limit CPU thread overhead for this small network.


class QNetwork(nn.Module):  # Define the neural network approximating Q(s, a).
    def __init__(self, state_dim, action_dim):  # Receive state and action dimensions.
        super().__init__()  # Initialize the base PyTorch module.
        self.layers = nn.Sequential(  # Stack layers in forward-pass order.
            nn.Linear(state_dim, 64),  # Convert the 4-dimensional state to 64 features.
            nn.ReLU(),  # Add a nonlinear activation.
            nn.Linear(64, 64),  # Add another hidden layer with 64 neurons.
            nn.ReLU(),  # Add another nonlinear activation.
            nn.Linear(64, action_dim),  # Predict a Q-value for each discrete action.
        )  # Finish the sequential network.

    def forward(self, states):  # Define how states pass through the network.
        return self.layers(states)  # Return one Q-value per action for each state.


class ReplayBuffer:  # Hold past experience so DQN can learn off-policy.
    def __init__(self, capacity):  # Initialize the memory with a maximum size.
        self.buffer = deque(maxlen=capacity)  # Automatically discard the oldest samples.

    def add(self, state, action, reward, next_state, terminated):  # Record a transition.
        self.buffer.append((state, action, reward, next_state, terminated))  # Save transition.

    def sample(self, batch_size):  # Randomly draw a minibatch from memory.
        batch = random.sample(self.buffer, batch_size)  # Sample without replacement.
        states, actions, rewards, next_states, terminals = zip(*batch)  # Split fields.
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)  # States tensor.
        actions = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)  # Action indices.
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # Rewards.
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=DEVICE)  # Next states.
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # Terminal flags.
        return states, actions, rewards, next_states, terminals  # Return training tensors.

    def __len__(self):  # Let len(buffer) return the number of stored transitions.
        return len(self.buffer)  # Return current replay-buffer size.


def epsilon_by_step(step):  # Compute the exploration rate for a given environment step.
    progress = min(step / EPSILON_DECAY_STEPS, 1.0)  # Clip progress to [0, 1].
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)  # Linearly decay epsilon.


def select_action(network, state, epsilon, env):  # Choose an epsilon-greedy action.
    if random.random() < epsilon:  # Explore with probability epsilon.
        return env.action_space.sample()  # Select one of the discrete actions randomly.
    with torch.no_grad():  # Disable gradients during action selection.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # Add batch dimension.
        q_values = network(state_tensor)  # Estimate Q-values for both actions.
        return int(q_values.argmax(dim=1).item())  # Select the action with the largest Q-value.


def optimize_dqn(online, target, optimizer, memory):  # Perform one Bellman-error update.
    states, actions, rewards, next_states, terminals = memory.sample(BATCH_SIZE)  # Sample replay transitions.
    chosen_q = online(states).gather(1, actions)  # Extract Q(s, a) for actions actually taken.
    with torch.no_grad():  # Stop gradients from entering the target network.
        next_q = target(next_states).max(dim=1, keepdim=True).values  # Best next-action value.
        td_target = rewards + GAMMA * (1.0 - terminals) * next_q  # One-step Bellman target.
    loss = nn.functional.smooth_l1_loss(chosen_q, td_target)  # Huber loss reduces outlier influence.
    optimizer.zero_grad()  # Remove gradients from the previous update.
    loss.backward()  # Differentiate the loss with respect to online-network weights.
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)  # Prevent extreme gradients.
    optimizer.step()  # Update the online Q-network parameters.
    return float(loss.item())  # Return scalar loss for optional monitoring.


def evaluate(network, episodes=5):  # Evaluate with a greedy policy (no exploration).
    env = gym.make('CartPole-v1')  # Create a separate evaluation environment.
    rewards = []  # Collect the return from each evaluation episode.
    for episode in range(episodes):  # Repeat evaluation several times.
        state, _ = env.reset(seed=SEED + 10_000 + episode)  # Set distinct deterministic resets.
        total_reward = 0.0  # Initialize episode return.
        for _ in range(MAX_STEPS):  # Take at most 500 actions.
            action = select_action(network, state, epsilon=0.0, env=env)  # Choose greedily.
            state, reward, terminated, truncated, _ = env.step(action)  # Step the environment.
            total_reward += reward  # Accumulate reward.
            if terminated or truncated:  # Stop at either kind of episode ending.
                break  # Exit this evaluation episode.
        rewards.append(total_reward)  # Save this episode's return.
    env.close()  # Release evaluation resources.
    return float(np.mean(rewards))  # Average across evaluation episodes.


def train():  # Run the complete DQN training procedure.
    env = gym.make('CartPole-v1')  # Create the training environment.
    env.action_space.seed(SEED)  # Seed random sampled actions.
    state_dim = env.observation_space.shape[0]  # CartPole states have four features.
    action_dim = env.action_space.n  # CartPole has two discrete actions.
    online = QNetwork(state_dim, action_dim).to(DEVICE)  # Create the trainable Q-network.
    target = QNetwork(state_dim, action_dim).to(DEVICE)  # Create a separate target Q-network.
    target.load_state_dict(online.state_dict())  # Start the target with identical weights.
    target.eval()  # Target network is only used for inference.
    optimizer = optim.Adam(online.parameters(), lr=LEARNING_RATE)  # Set up Adam optimizer.
    memory = ReplayBuffer(BUFFER_SIZE)  # Create experience replay memory.
    recent_returns = deque(maxlen=20)  # Track the last twenty episode returns.
    global_step = 0  # Count interactions with the environment.
    gradient_step = 0  # Count optimizer updates.

    for episode in range(1, EPISODES + 1):  # Loop through training episodes.
        state, _ = env.reset(seed=SEED + episode)  # Start a fresh episode.
        episode_return = 0.0  # Reset the reward total.
        for _ in range(MAX_STEPS):  # Interact for at most 500 steps.
            epsilon = epsilon_by_step(global_step)  # Determine current exploration rate.
            action = select_action(online, state, epsilon, env)  # Choose exploratory or greedy action.
            next_state, reward, terminated, truncated, _ = env.step(action)  # Apply action.
            memory.add(state, action, reward, next_state, terminated)  # Store transition.
            state = next_state  # Continue from the new state.
            episode_return += reward  # Record this time step's reward.
            global_step += 1  # Advance the environment-step counter.

            if len(memory) >= MIN_REPLAY_SIZE:  # Only learn after sufficient experience exists.
                optimize_dqn(online, target, optimizer, memory)  # Update online Q-values.
                gradient_step += 1  # Record one optimizer step.
                if gradient_step % TARGET_UPDATE_INTERVAL == 0:  # Periodically update target weights.
                    target.load_state_dict(online.state_dict())  # Hard-copy online parameters.

            if terminated or truncated:  # End the episode if failure or time limit occurs.
                break  # Start another episode.

        recent_returns.append(episode_return)  # Store the completed episode's return.
        if episode % 10 == 0:  # Report training progress every ten episodes.
            average_return = np.mean(recent_returns)  # Calculate recent mean reward.
            print(f'Episode {episode:3d} | Steps {global_step:6d} | '
                  f'Mean return (last 20) {average_return:6.1f} | Epsilon {epsilon:.3f}')  # Print metrics.

    env.close()  # Close training environment.
    torch.save(online.state_dict(), 'dqn_cartpole.pt')  # Save learned model parameters.
    print('Saved model to dqn_cartpole.pt')  # Confirm where the model was saved.
    score = evaluate(online, episodes=5)  # Run greedy evaluation over five episodes.
    print(f'Mean evaluation reward over 5 episodes: {score:.1f} / 500')  # Show test result.
    return online  # Return the trained model for further experiments.


if __name__ == '__main__':  # Only run training when executing this file directly.
    train()  # Start DQN training.
