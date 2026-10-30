"""CPU-friendly Soft Actor-Critic (SAC) mobile robot navigation demo.
Install: pip install torch numpy matplotlib
Run: python sac_robot_navigation.py --episodes 100
"""
import argparse
import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Use a single CPU thread to keep tiny-network training efficient on laptops.
torch.set_num_threads(1)


def seed_all(seed):
    # Seed Python's random generator.
    random.seed(seed)
    # Seed NumPy's random generator.
    np.random.seed(seed)
    # Seed PyTorch's random generator.
    torch.manual_seed(seed)


class RobotEnv:
    """Differential-drive-like robot with velocity and turn-rate commands."""

    def __init__(self):
        # Store the width and height of the square environment.
        self.world_size = 5.0
        # Define a stationary circular obstacle as x, y, radius.
        self.obstacle = (2.5, 2.5, 0.65)
        # Set the robot's collision radius.
        self.robot_radius = 0.12
        # Define the navigation goal.
        self.goal = np.array([4.3, 4.3], dtype=np.float32)
        # Simulation time step in seconds.
        self.dt = 0.20
        # Maximum number of simulator actions in each episode.
        self.max_steps = 110
        # Initialize robot state through reset().
        self.reset()

    def reset(self):
        # Start with a little randomized position and heading.
        self.x = float(0.65 + np.random.uniform(-0.15, 0.15))
        self.y = float(0.65 + np.random.uniform(-0.15, 0.15))
        self.theta = float(np.random.uniform(-0.35, 0.35))
        # Reset episode step counter.
        self.steps = 0
        # Save the robot's trajectory for visualization.
        self.path = [(self.x, self.y)]
        # Return the initial observation vector.
        return self._obs()

    def _obs(self):
        # Calculate goal direction relative to the robot.
        delta = self.goal - np.array([self.x, self.y])
        # Calculate obstacle direction relative to the robot.
        obstacle_delta = np.array(self.obstacle[:2]) - np.array([self.x, self.y])
        # Normalize features to moderate numerical scales.
        return np.array([
            self.x / self.world_size,
            self.y / self.world_size,
            math.cos(self.theta),
            math.sin(self.theta),
            delta[0] / self.world_size,
            delta[1] / self.world_size,
            obstacle_delta[0] / self.world_size,
            obstacle_delta[1] / self.world_size,
        ], dtype=np.float32)

    def step(self, action):
        # Ensure actions lie in [-1, 1] before converting to motor commands.
        action = np.clip(np.asarray(action), -1.0, 1.0)
        # Map the first control to forward speed in [0, 1] meters/sec.
        speed = 0.5 * (float(action[0]) + 1.0)
        # Map the second control to angular velocity in [-2, 2] rad/sec.
        omega = 2.0 * float(action[1])
        # Calculate distance to the target before moving.
        old_distance = float(np.linalg.norm(self.goal - [self.x, self.y]))
        # Update the robot heading using the commanded angular velocity.
        self.theta += omega * self.dt
        # Update position using the simple unicycle motion model.
        self.x += speed * math.cos(self.theta) * self.dt
        self.y += speed * math.sin(self.theta) * self.dt
        # Increment elapsed simulation steps.
        self.steps += 1
        # Save the new position for trajectory plotting.
        self.path.append((self.x, self.y))
        # Measure the remaining distance to the goal.
        distance = float(np.linalg.norm(self.goal - [self.x, self.y]))
        # Measure the robot's distance to the obstacle center.
        obstacle_distance = math.hypot(self.x-self.obstacle[0], self.y-self.obstacle[1])
        # Detect intersection between robot and obstacle.
        collision = obstacle_distance <= self.obstacle[2] + self.robot_radius
        # Detect crossing the world boundaries.
        outside = not (0.0 <= self.x <= self.world_size and 0.0 <= self.y <= self.world_size)
        # Consider reaching within 0.35m of the goal a success.
        success = distance < 0.35
        # Give reward for approaching the target (potential-based progress).
        reward = 10.0 * (old_distance - distance)
        # Impose a small time penalty to encourage efficient paths.
        reward -= 0.03
        # Penalize going close to the obstacle, even before a collision.
        clearance = obstacle_distance - self.obstacle[2] - self.robot_radius
        if clearance < 0.35:
            reward -= 0.08 * (0.35 - clearance) / 0.35
        # Strongly penalize collisions and invalid positions.
        if collision or outside:
            reward -= 15.0
        # Reward successful goal reaching.
        if success:
            reward += 25.0
        # A terminal state is success, collision, or leaving the workspace.
        terminated = success or collision or outside
        # Time limit is a separate truncation, not an MDP termination.
        truncated = self.steps >= self.max_steps and not terminated
        # Return observation, reward, termination, truncation, diagnostics.
        return self._obs(), float(reward), terminated, truncated, {"success": success, "collision": collision or outside}


class ReplayBuffer:
    def __init__(self, capacity):
        # Store up to capacity previously observed transitions.
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        # Copy transition arrays so later environment updates cannot alter them.
        self.data.append((state.copy(), action.copy(), reward, next_state.copy(), terminated))

    def sample(self, batch_size):
        # Uniformly sample transitions without replacement.
        batch = random.sample(self.data, batch_size)
        # Reorganize tuples into individual arrays.
        states, actions, rewards, next_states, terminated = zip(*batch)
        # Return batched torch tensors on the CPU.
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.as_tensor(np.asarray(terminated), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        # Report available number of transitions.
        return len(self.data)


def mlp(input_dim, output_dim):
    # Build a small CPU-friendly two-hidden-layer neural network.
    return nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, output_dim))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        # Initialize the neural-network base class.
        super().__init__()
        # Predict mean and log-standard-deviation for Gaussian policy.
        self.net = mlp(obs_dim, 2 * act_dim)

    def sample(self, states, deterministic=False):
        # Calculate raw Gaussian parameters from the observation.
        mean, log_std = self.net(states).chunk(2, dim=-1)
        # Prevent extreme variances that cause numerical problems.
        log_std = torch.clamp(log_std, -5.0, 2.0)
        # Convert log standard deviation to standard deviation.
        std = log_std.exp()
        # Create the Gaussian distribution in unconstrained action space.
        dist = torch.distributions.Normal(mean, std)
        # Draw differentiable samples using the reparameterization trick.
        raw = mean if deterministic else dist.rsample()
        # Squash Gaussian values into allowed action range [-1, 1].
        action = torch.tanh(raw)
        # Correct log probability for the tanh transformation (Jacobian).
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        # Sum over the two control dimensions to get one log probability.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Return action and its corrected log probability.
        return action, log_prob


class SAC:
    def __init__(self, obs_dim=8, act_dim=2, gamma=0.99, tau=0.005):
        # Discount factor for future rewards.
        self.gamma = gamma
        # Polyak interpolation factor for slow target updates.
        self.tau = tau
        # Create a stochastic actor policy network.
        self.actor = Actor(obs_dim, act_dim)
        # Create two separate Q-functions to reduce overestimation bias.
        self.q1 = mlp(obs_dim + act_dim, 1)
        self.q2 = mlp(obs_dim + act_dim, 1)
        # Create corresponding slowly updated target Q-networks.
        self.q1_target = mlp(obs_dim + act_dim, 1)
        self.q2_target = mlp(obs_dim + act_dim, 1)
        # Initialize both target networks using online Q-network weights.
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # Set up the actor optimization algorithm.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        # Set up the Q-function optimization algorithm.
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4)
        # Optimize log(alpha) to keep temperature strictly positive.
        self.log_alpha = torch.tensor(math.log(0.2), requires_grad=True)
        # Set up automatic entropy-temperature optimization.
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        # Common SAC target entropy heuristic: negative action dimension.
        self.target_entropy = -float(act_dim)

    def act(self, obs, deterministic=False):
        # Disable gradient tracking during environmental interaction.
        with torch.no_grad():
            # Add a batch dimension to the robot's observation.
            s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            # Sample a bounded continuous control from the policy.
            a, _ = self.actor.sample(s, deterministic)
            # Return a NumPy action for the simulator.
            return a.squeeze(0).numpy()

    def update(self, buffer, batch_size):
        # Sample a batch of past interactions.
        s, a, r, ns, terminal = buffer.sample(batch_size)
        # Detach alpha for updates that must not optimize its parameter.
        alpha = self.log_alpha.exp().detach()

        # --- Twin-critic Bellman target ---
        with torch.no_grad():
            # Sample the next action from the current stochastic actor.
            na, next_logp = self.actor.sample(ns)
            # Concatenate next state and next action for the critics.
            next_sa = torch.cat([ns, na], dim=-1)
            # Take the smaller value estimate to limit Q overestimation.
            min_next_q = torch.min(self.q1_target(next_sa), self.q2_target(next_sa))
            # Add entropy reward to the bootstrapped target value.
            next_value = min_next_q - alpha * next_logp
            # Do not bootstrap through true terminal transitions.
            target = r + self.gamma * (1 - terminal) * next_value

        # Build the online critic input using sampled stored actions.
        sa = torch.cat([s, a], dim=-1)
        # Use mean squared Bellman errors for both critics.
        q_loss = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        # Clear existing critic gradients.
        self.q_opt.zero_grad()
        # Backpropagate Bellman errors through critic networks.
        q_loss.backward()
        # Update both Q-functions.
        self.q_opt.step()

        # --- Actor loss ---
        # Sample policy actions at the replayed states.
        new_actions, logp = self.actor.sample(s)
        # Evaluate those actions with the latest critics.
        policy_sa = torch.cat([s, new_actions], dim=-1)
        # SAC maximizes min(Q1,Q2) plus entropy; minimize its negative.
        actor_loss = (alpha * logp - torch.min(self.q1(policy_sa), self.q2(policy_sa))).mean()
        # Reset actor gradients.
        self.actor_opt.zero_grad()
        # Compute actor gradients through action reparameterization.
        actor_loss.backward()
        # Update the policy network weights.
        self.actor_opt.step()

        # --- Automatic entropy temperature ---
        # Increase alpha when policy entropy is below the target.
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        # Reset temperature gradients.
        self.alpha_opt.zero_grad()
        # Differentiate the temperature loss.
        alpha_loss.backward()
        # Update log(alpha).
        self.alpha_opt.step()

        # --- Slowly update target critics ---
        with torch.no_grad():
            for target_net, online_net in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
                for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                    # Polyak update prevents rapid target changes.
                    target_param.mul_(1 - self.tau).add_(self.tau * online_param)
        # Return metrics for optional logging.
        return float(q_loss.item()), float(actor_loss.item()), float(self.log_alpha.exp().item())


def evaluate(agent, episodes=10):
    # Create an independent evaluation environment.
    env = RobotEnv()
    # Track how many episodes reach the goal.
    wins = 0
    # Save a sample trajectory for plotting.
    example_path = []
    # Evaluate without exploration noise from the actor.
    for episode in range(episodes):
        obs = env.reset()
        for step in range(env.max_steps):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        wins += int(info["success"])
        if episode == 0:
            example_path = list(env.path)
    # Return empirical success rate and example path.
    return wins / episodes, example_path


def plot_results(env, rewards, path, out_file):
    # Create two plots in a single PNG file.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Plot a noisy learning curve and its moving average.
    axes[0].plot(rewards, alpha=0.35, label="Episode reward")
    window = min(15, max(1, len(rewards)))
    averaged = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(np.arange(window - 1, len(rewards)), averaged, label="Moving average")
    axes[0].set(xlabel="Episode", ylabel="Total reward", title="SAC learning curve")
    axes[0].legend()
    # Draw the target as a green circle.
    goal_circle = plt.Circle(tuple(env.goal), 0.35, color="green", alpha=0.25)
    axes[1].add_patch(goal_circle)
    # Draw the obstacle and robot clearance footprint.
    obstacle_circle = plt.Circle(env.obstacle[:2], env.obstacle[2], color="firebrick", alpha=0.5)
    axes[1].add_patch(obstacle_circle)
    # Draw the navigation trajectory if one was supplied.
    trajectory = np.asarray(path, dtype=np.float32)
    if len(trajectory) > 0:
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], label="Robot path")
        axes[1].scatter(*trajectory[0], marker="o", label="Start")
        axes[1].scatter(*trajectory[-1], marker="x", label="End")
    axes[1].set(xlim=(0, env.world_size), ylim=(0, env.world_size),
                xlabel="X (m)", ylabel="Y (m)", title="Robot navigation")
    axes[1].set_aspect("equal")
    axes[1].legend(loc="best")
    # Prevent the two panels from overlapping.
    fig.tight_layout()
    # Save the visualized result locally.
    fig.savefig(out_file, dpi=140)
    # Close the figure to conserve memory.
    plt.close(fig)


def main():
    # Define command-line flags for laptop-friendly run settings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot", type=str, default="sac_robot_results.png")
    args = parser.parse_args()
    # Apply random seeds for more reproducible learning.
    seed_all(args.seed)
    # Create the 2D robot simulator and SAC agent.
    env = RobotEnv()
    agent = SAC()
    # Store past robot experiences in an off-policy replay buffer.
    buffer = ReplayBuffer(capacity=100000)
    # Track total number of environment interactions.
    total_steps = 0
    # Track episode rewards for plotting training progress.
    rewards = []
    # Begin training episodes.
    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0.0
        info = {"success": False}
        for step in range(env.max_steps):
            # At first, collect diverse transitions with uniform actions.
            action = np.random.uniform(-1, 1, size=2).astype(np.float32) if total_steps < args.warmup else agent.act(obs)
            # Apply the action to the robot simulator.
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Store transition; importantly, use termination, not truncation.
            buffer.add(obs, action, reward, next_obs, terminated)
            # Train SAC after enough experiences have been accumulated.
            if total_steps >= args.warmup and len(buffer) >= args.batch_size:
                agent.update(buffer, args.batch_size)
            # Move to the next observation.
            obs = next_obs
            # Accumulate the total episode reward.
            episode_reward += reward
            # Increment global environment-step counter.
            total_steps += 1
            # Reset episode if the robot reached a terminal condition.
            if terminated or truncated:
                break
        # Record reward for later plotting.
        rewards.append(episode_reward)
        # Print short, occasional training progress updates.
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode+1:3d}/{args.episodes} | reward {episode_reward:7.2f} | "
                  f"success={info['success']} | steps={step+1} | replay={len(buffer)}")
    # Evaluate performance after training using deterministic controls.
    success_rate, path = evaluate(agent, episodes=10)
    print(f"Evaluation success rate: {success_rate:.0%} (10 trials)")
    # Save model weights for future use.
    torch.save(agent.actor.state_dict(), "sac_robot_actor.pt")
    # Save the learning curve and an example robot trajectory.
    plot_results(env, rewards, path, args.plot)
    print(f"Saved plot to {args.plot} and actor weights to sac_robot_actor.pt")


if __name__ == "__main__":
    # Run the training script when executed directly.
    main()
