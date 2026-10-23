"""Laptop-scale robot RL research: offline data, videos without actions, sparse rewards, world models
 and reproducible comparisons."""
import argparse  # Parse reproducible command-line experiment settings.
import csv  # Save experiment results to a spreadsheet-compatible format.
import json  # Save experiment metadata and per-run metrics.
import random  # Seed Python randomness and choose replay samples.
from dataclasses import dataclass  # Represent transitions clearly.
from pathlib import Path  # Create portable experiment output paths.
import numpy as np  # Simulate two-dimensional robot dynamics.
import torch  # Build and train all neural networks on the CPU.
from torch import nn  # Neural-network layers and loss functions.
from torch.nn import functional as F  # Neural-network operations.

# Limit CPU thread contention on laptops with multiple cores.
torch.set_num_threads(1)


def seed_everything(seed):  # Reproduce initializations, datasets, and training batches.
    random.seed(seed)  # Initialize the Python pseudorandom generator.
    np.random.seed(seed)  # Initialize NumPy's pseudorandom generator.
    torch.manual_seed(seed)  # Initialize PyTorch's pseudorandom generator.


class PointRobot:  # Small continuous-control 2D point robot.
    def __init__(self, horizon=35):  # Set the maximum number of control steps.
        self.horizon = horizon  # Store episode horizon.
        self.goal = np.array([0.8, 0.8], dtype=np.float32)  # Fixed target position.
        self.state = np.zeros(4, dtype=np.float32)  # [x, y, velocity_x, velocity_y].
        self.t = 0  # Current step of the episode.

    def reset(self, rng):  # Sample a new initial condition.
        self.t = 0  # Reset the time counter.
        self.state = np.array([rng.uniform(-0.9, -0.5), rng.uniform(-0.9, -0.5), 0, 0], dtype=np.float32)
        return self.state.copy()  # Return a copy to avoid accidental mutations.

    def step(self, action):  # Apply bounded two-dimensional acceleration.
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)  # Enforce actuator limits.
        self.state[2:] = 0.85 * self.state[2:] + 0.15 * a  # Damped velocity dynamics.
        self.state[:2] = self.state[:2] + 0.18 * self.state[2:]  # Integrate position.
        self.state[:2] = np.clip(self.state[:2], -1.2, 1.2)  # Keep robot inside arena.
        self.t += 1  # Advance simulation time.
        reached = np.linalg.norm(self.state[:2] - self.goal) < 0.16  # Test success radius.
        terminated = bool(reached)  # Mark true task termination.
        truncated = self.t >= self.horizon  # Mark time-limit truncation separately.
        reward = 1.0 if reached else 0.0  # Reward only upon reaching the goal.
        return self.state.copy(), reward, terminated, truncated  # Return Gym-like results.


def expert_action(state, goal):  # Simple proportional-derivative expert controller.
    direction = goal - state[:2]  # Point toward the goal.
    action = 3.0 * direction - 1.0 * state[2:]  # Compensate for momentum.
    return np.clip(action, -1, 1).astype(np.float32)  # Respect action bounds.


@dataclass
class Dataset:  # Container for offline transitions and observation-only videos.
    s: np.ndarray  # Current robot states.
    a: np.ndarray  # Recorded actions; absent for video-only data.
    r: np.ndarray  # Sparse recorded rewards.
    sp: np.ndarray  # Next robot states.
    done: np.ndarray  # True terminal flags (not time limits).
    quality: np.ndarray  # Source identifiers for mixed-quality datasets.
    videos: list  # Lists of state sequences with deliberately no action labels.


def collect_data(seed, episodes=90, video_episodes=30, horizon=35):  # Build mixed-quality data.
    rng = np.random.default_rng(seed)  # Use an independent deterministic data RNG.
    env = PointRobot(horizon=horizon)  # Initialize robot simulator.
    rows = []  # Store action-labeled transitions.
    videos = []  # Store observation-only sequences.
    for episode in range(episodes + video_episodes):  # Produce both data sources.
        obs = env.reset(rng)  # Reset the robot.
        observations = [obs.copy()]  # Begin a video sequence with its first state.
        is_video = episode >= episodes  # Last episodes omit all action labels.
        quality = episode % 3  # Mix expert, noisy, and random trajectories.
        for _ in range(horizon):  # Step until success or timeout.
            expert = expert_action(obs, env.goal)  # Compute privileged demonstration.
            if quality == 0:  # High-quality data source.
                action = expert  # Execute expert controls.
            elif quality == 1:  # Medium-quality data source.
                action = np.clip(expert + rng.normal(0, 0.7, 2), -1, 1)  # Add motor noise.
            else:  # Low-quality source.
                action = rng.uniform(-1, 1, 2)  # Act randomly.
            nxt, rew, done, truncated = env.step(action)  # Advance simulator.
            observations.append(nxt.copy())  # Record observation sequence (the "video").
            if not is_video:  # Store full transitions only for action-labeled data.
                rows.append((obs.copy(), action.copy(), rew, nxt.copy(), float(done), quality))
            obs = nxt  # Advance the current state.
            if done or truncated:  # End completed or timed-out episodes.
                break  # Exit the trajectory loop.
        if is_video:  # Deliberately discard actions from video trajectories.
            videos.append(np.stack(observations).astype(np.float32))  # Keep states only.
    s, a, r, sp, done, quality = zip(*rows)  # Separate fields for efficient sampling.
    return Dataset(np.stack(s), np.stack(a), np.array(r, dtype=np.float32)[:, None],
                   np.stack(sp), np.array(done, dtype=np.float32)[:, None],
                   np.array(quality, dtype=np.int64), videos)  # Return offline corpus.


def mlp(in_dim, out_dim, hidden):  # Construct a small trainable multilayer perceptron.
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, out_dim))  # Two hidden layers.


class InverseDynamics(nn.Module):  # Infer missing actions from successive observations.
    def __init__(self, hidden):  # Configure model capacity.
        super().__init__()  # Register trainable parameters with PyTorch.
        self.net = mlp(8, 2, hidden)  # [state,next_state] -> estimated action.

    def forward(self, s, sp):  # Estimate action from a state transition.
        return torch.tanh(self.net(torch.cat([s, sp], dim=-1)))  # Keep estimates in [-1,1].


class WorldModel(nn.Module):  # Learn dynamics p(s_next | s, action) by regression.
    def __init__(self, hidden):  # Define lightweight dynamics network.
        super().__init__()  # Initialize nn.Module.
        self.net = mlp(6, 4, hidden)  # Predict a four-dimensional state difference.

    def forward(self, s, a):  # Predict next state using residual dynamics.
        return s + self.net(torch.cat([s, a], dim=-1))  # Residual predictions ease fitting.


class QNetwork(nn.Module):  # Learn sparse-reward action values for policy evaluation.
    def __init__(self, hidden):  # Configure critic size.
        super().__init__()  # Initialize model internals.
        self.net = mlp(6, 1, hidden)  # Input state and action, output a Q-value.

    def forward(self, s, a):  # Evaluate actions at robot states.
        return self.net(torch.cat([s, a], dim=-1))  # Concatenate state and action.


class Policy(nn.Module):  # Deterministic robot actor learned by behavior cloning.
    def __init__(self, hidden):  # Select network width.
        super().__init__()  # Initialize PyTorch module.
        self.net = mlp(4, 2, hidden)  # Map robot state to 2D control signal.

    def forward(self, s):  # Produce bounded motor commands.
        return torch.tanh(self.net(s))  # Bound each component to [-1,1].


def to_tensor(array):  # Convert NumPy training arrays into CPU float tensors.
    return torch.as_tensor(array, dtype=torch.float32)  # No GPU allocations.


def train_inverse(data, hidden, steps, seed):  # Fit inverse dynamics only on labeled data.
    torch.manual_seed(seed + 100)  # Make model initialization deterministic.
    model = InverseDynamics(hidden)  # Create inverse-dynamics network.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # Set regression optimizer.
    s, sp, a = map(to_tensor, (data.s, data.sp, data.a))  # Cache tensors.
    rng = np.random.default_rng(seed + 101)  # Sample stable minibatches.
    for _ in range(steps):  # Run supervised inverse-model training.
        ids = rng.integers(len(s), size=128)  # Draw labeled transitions.
        loss = F.mse_loss(model(s[ids], sp[ids]), a[ids])  # Match recorded actions.
        opt.zero_grad()  # Reset previous gradients.
        loss.backward()  # Differentiate the mean-square loss.
        opt.step()  # Update inverse-model weights.
    return model  # Return learned pseudo-labeling model.


def build_pseudo_labeled(data, inverse):  # Convert observation-only videos to action estimates.
    states, actions, next_states = [], [], []  # Collect synthetic transitions.
    with torch.no_grad():  # Disable gradients during pseudo-label generation.
        for seq in data.videos:  # Process one video trajectory at a time.
            if len(seq) < 2:  # Ignore empty transitions.
                continue  # Skip trajectories too short for inference.
            s, sp = seq[:-1], seq[1:]  # Extract all adjacent observation pairs.
            estimated = inverse(to_tensor(s), to_tensor(sp)).numpy()  # Infer action labels.
            states.append(s)  # Add input states.
            actions.append(estimated)  # Add inferred actions, not true recorded actions.
            next_states.append(sp)  # Add next states.
    return (np.concatenate(states), np.concatenate(actions), np.concatenate(next_states)) if states else None


def behavior_clone(data, hidden, steps, seed, mode="expert", pseudo=None):  # Compare data regimes.
    torch.manual_seed(seed + 200)  # Seed actor initialization.
    model = Policy(hidden)  # Initialize actor.
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)  # Initialize actor optimizer.
    if mode == "expert":  # Strong labeled baseline: only clean demonstrations.
        selected = data.quality == 0  # Filter clean source examples.
        states, actions = data.s[selected], data.a[selected]  # Extract clean pairs.
    else:  # Mixed-source setting.
        states, actions = data.s, data.a  # Start with all labeled data.
        if mode == "mixed_video" and pseudo is not None:  # Add unlabeled video pseudo-actions.
            states = np.concatenate([states, pseudo[0]])  # Extend state inputs.
            actions = np.concatenate([actions, pseudo[1]])  # Extend action targets.
    s, a = to_tensor(states), to_tensor(actions)  # Move arrays into tensors.
    rng = np.random.default_rng(seed + 201)  # Initialize mini-batch sampler.
    for _ in range(steps):  # Perform behavior cloning updates.
        ids = rng.integers(len(s), size=128)  # Draw a mini-batch.
        loss = F.mse_loss(model(s[ids]), a[ids])  # Imitate dataset actions.
        opt.zero_grad()  # Zero parameter gradients.
        loss.backward()  # Differentiate cloning objective.
        opt.step()  # Update actor parameters.
    return model  # Return behavior-cloned policy.


def train_world_model(data, hidden, steps, seed, pseudo=None):  # Fit data-driven dynamics.
    torch.manual_seed(seed + 300)  # Reproduce initial weights.
    model = WorldModel(hidden)  # Construct learned world model.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimize dynamics MSE.
    s, a, sp = data.s, data.a, data.sp  # Read labeled transitions.
    if pseudo is not None:  # Include pseudo-labeled observation-only transitions.
        s = np.concatenate([s, pseudo[0]])  # Extend state data.
        a = np.concatenate([a, pseudo[1]])  # Extend action data.
        sp = np.concatenate([sp, pseudo[2]])  # Extend next-state data.
    s, a, sp = map(to_tensor, (s, a, sp))  # Convert to tensors.
    rng = np.random.default_rng(seed + 301)  # Seed random mini-batch selection.
    for _ in range(steps):  # Fit the world model.
        ids = rng.integers(len(s), size=128)  # Sample transitions with replacement.
        loss = F.mse_loss(model(s[ids], a[ids]), sp[ids])  # Minimize prediction error.
        opt.zero_grad()  # Clear accumulated gradients.
        loss.backward()  # Backpropagate model loss.
        opt.step()  # Update dynamics parameters.
    return model  # Return trained simulator approximation.


def train_sparse_q(data, hidden, steps, seed, gamma=0.98):  # Critic learning from sparse rewards.
    torch.manual_seed(seed + 400)  # Reproduce network initialization.
    critic = QNetwork(hidden)  # Build value estimator.
    target = QNetwork(hidden)  # Build target estimator for stable TD targets.
    target.load_state_dict(critic.state_dict())  # Synchronize initial parameters.
    opt = torch.optim.Adam(critic.parameters(), lr=8e-4)  # Configure Q optimizer.
    s, a, r, sp, done = map(to_tensor, (data.s, data.a, data.r, data.sp, data.done))
    rng = np.random.default_rng(seed + 401)  # Set TD minibatch RNG.
    goal = np.array([0.8, 0.8], dtype=np.float32)  # Known target for demonstration.
    for step in range(steps):  # Perform fitted temporal-difference updates.
        ids = rng.integers(len(s), size=128)  # Sample offline transitions.
        with torch.no_grad():  # Do not backpropagate through Bellman target.
            next_action = to_tensor(np.stack([expert_action(x.numpy(), goal) for x in sp[ids]]))
            bootstrap = target(sp[ids], next_action)  # Evaluate fixed expert for bootstrapping.
            y = r[ids] + gamma * (1.0 - done[ids]) * bootstrap  # Sparse-reward TD target.
        loss = F.smooth_l1_loss(critic(s[ids], a[ids]), y)  # Robust Bellman-error loss.
        opt.zero_grad()  # Clear critic gradients.
        loss.backward()  # Differentiate Bellman residual.
        opt.step()  # Update Q-values.
        if (step + 1) % 50 == 0:  # Update target critic periodically.
            target.load_state_dict(critic.state_dict())  # Hard-copy stable target.
    return critic  # Return learned sparse-reward value function.


def mpc_action(state, world, horizon=7, candidates=80, seed=0):  # Random-shooting model-predictive controller.
    rng = np.random.default_rng(seed)  # Reproducible candidate controls.
    candidate_actions = rng.uniform(-1, 1, size=(candidates, horizon, 2)).astype(np.float32)
    candidate_actions[0] = 0  # Include a zero-control candidate for safety comparison.
    sim_states = to_tensor(np.repeat(state[None, :], candidates, axis=0))  # Batch start states.
    with torch.no_grad():  # Planning does not modify world-model weights.
        for t in range(horizon):  # Imagine each candidate's predicted trajectory.
            sim_states = world(sim_states, to_tensor(candidate_actions[:, t, :]))  # Predict next states.
        goal = torch.tensor([0.8, 0.8])  # Desired terminal position.
        final_distance = ((sim_states[:, :2] - goal) ** 2).sum(dim=1)  # Terminal cost.
        control_cost = 0.01 * (to_tensor(candidate_actions) ** 2).sum(dim=(1, 2))  # Energy cost.
        best = torch.argmin(final_distance + control_cost).item()  # Select lowest-cost plan.
    return candidate_actions[best, 0]  # Apply only first action; replan at next state.


def evaluate(controller, seed, episodes=15, horizon=35):  # Independent held-out environment seeds.
    rng = np.random.default_rng(seed + 9000)  # Evaluation RNG cannot affect training RNG.
    env = PointRobot(horizon)  # Fresh evaluation environment.
    returns, lengths = [], []  # Track sparse goal success and episode lengths.
    for episode in range(episodes):  # Evaluate multiple initial states.
        state = env.reset(rng)  # Sample new initial state.
        total = 0.0  # Accumulate rewards.
        for t in range(horizon):  # Run a policy without gradient updates.
            action = controller(state, episode, t)  # Request control action.
            state, reward, done, truncated = env.step(action)  # Execute action.
            total += reward  # Track success reward.
            if done or truncated:  # Stop after success or timeout.
                break  # End current evaluation episode.
        returns.append(total)  # Store episode return.
        lengths.append(t + 1)  # Store elapsed steps.
    return {"success_rate": float(np.mean(returns)), "mean_length": float(np.mean(lengths))}


def evaluate_policy(policy, seed, episodes=15):  # Wrap a PyTorch actor as controller.
    def act(state, episode, t):  # Match evaluate() controller signature.
        with torch.no_grad():  # Disable gradient tracking in deployment.
            return policy(to_tensor(state[None, :]))[0].numpy()  # Convert action to NumPy.
    return evaluate(act, seed, episodes=episodes)  # Return held-out metrics.


def run_experiment(args, seed, hidden, size):  # One reproducible configuration.
    seed_everything(seed)  # Fix all RNG states.
    data = collect_data(seed, episodes=size, video_episodes=max(6, size // 3))  # Collect offline corpus.
    inverse = train_inverse(data, hidden, args.inverse_steps, seed)  # Train missing-action estimator.
    pseudo = build_pseudo_labeled(data, inverse)  # Infer actions for observation-only sequences.
    world = train_world_model(data, hidden, args.world_steps, seed, pseudo=pseudo)  # Fit world model.
    critic = train_sparse_q(data, hidden, args.critic_steps, seed)  # Learn sparse-reward Q estimates.
    results = []  # Accumulate baseline and method comparisons.
    for mode in ("expert", "mixed", "mixed_video"):  # Run dataset-quality ablations.
        actor = behavior_clone(data, hidden, args.bc_steps, seed, mode, pseudo)  # Train behavior policy.
        metrics = evaluate_policy(actor, seed)  # Evaluate on unseen initial conditions.
        results.append({"seed": seed, "width": hidden, "episodes": size, "method": mode, **metrics})
        if mode == "mixed_video":  # Save final checkpoint with model and configuration.
            torch.save({"actor": actor.state_dict(), "hidden": hidden, "seed": seed},
                       args.output / f"actor_seed{seed}_h{hidden}_n{size}.pt")
    def planner(state, episode, t):  # Wrap planning into evaluation controller interface.
        return mpc_action(state, world, horizon=args.planning_horizon,
                          candidates=args.planning_candidates, seed=seed * 100000 + episode * 1000 + t)
    planner_metrics = evaluate(planner, seed, episodes=8)  # Evaluate learned-model planner.
    results.append({"seed": seed, "width": hidden, "episodes": size, "method": "world_mpc", **planner_metrics})
    with torch.no_grad():  # Measure fitted one-step prediction and Q residuals.
        ids = slice(0, min(256, len(data.s)))  # Use a fixed diagnostic subset.
        model_mse = F.mse_loss(world(to_tensor(data.s[ids]), to_tensor(data.a[ids])),
                               to_tensor(data.sp[ids])).item()  # Dynamics prediction error.
        mean_q = critic(to_tensor(data.s[ids]), to_tensor(data.a[ids])).mean().item()  # Q diagnostic.
    for record in results:  # Attach measurements to every method configuration.
        record["world_model_mse"] = model_mse  # Quantify prediction fit on diagnostic sample.
        record["mean_offline_q"] = mean_q  # Record learned Q-value diagnostic.
        record["labeled_transitions"] = len(data.s)  # Record labeled dataset scale.
        record["video_transitions"] = sum(len(v) - 1 for v in data.videos)  # Count unlabeled transitions.
    return results  # Return experiment summaries.


def main():  # Main experiment runner and reproducibility entry point.
    parser = argparse.ArgumentParser(description="CPU robot RL research prototype")  # CLI help.
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test")  # Fast check.
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22])  # Replicate runs.
    parser.add_argument("--sizes", type=int, nargs="+", default=[60, 120])  # Data scaling axis.
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64])  # Network scaling axis.
    parser.add_argument("--output", type=Path, default=Path("robot_rl_outputs"))  # Output folder.
    parser.add_argument("--inverse-steps", type=int, default=160)  # Inverse learning budget.
    parser.add_argument("--world-steps", type=int, default=250)  # Dynamics learning budget.
    parser.add_argument("--critic-steps", type=int, default=200)  # TD learning budget.
    parser.add_argument("--bc-steps", type=int, default=200)  # Policy cloning budget.
    parser.add_argument("--planning-horizon", type=int, default=7)  # MPC horizon.
    parser.add_argument("--planning-candidates", type=int, default=80)  # MPC shooting samples.
    args = parser.parse_args()  # Read command-line settings.
    if args.quick:  # Minimize the budget for a sanity check.
        args.seeds, args.sizes, args.widths = [11], [24], [32]  # One small configuration.
        args.inverse_steps = args.world_steps = args.critic_steps = args.bc_steps = 35  # Short fits.
        args.planning_candidates = 25  # Cheap trajectory search.
    args.output.mkdir(parents=True, exist_ok=True)  # Create output directory.
    records = []  # Store all experimental metrics.
    for seed in args.seeds:  # Repeat for multiple independent random seeds.
        for size in args.sizes:  # Test increasing dataset sizes.
            for width in args.widths:  # Test increasing model sizes.
                print(f"Running seed={seed}, episodes={size}, hidden={width}", flush=True)  # Track progress.
                rows = run_experiment(args, seed, width, size)  # Run configuration.
                records.extend(rows)  # Append baseline and method metrics.
                for row in rows:  # Display each method's evaluation result.
                    print(f"  {row['method']:12s} success={row['success_rate']:.3f} ",
                          f"mean_steps={row['mean_length']:.1f}", flush=True)  # Compact metrics.
    path = args.output / "results.csv"  # Results table path.
    with path.open("w", newline="", encoding="utf-8") as file:  # Create CSV file.
        writer = csv.DictWriter(file, fieldnames=list(records[0]))  # Set column names.
        writer.writeheader()  # Write CSV header.
        writer.writerows(records)  # Save every experiment row.
    metadata = vars(args).copy()  # Preserve complete experiment configuration.
    metadata["output"] = str(metadata["output"])  # Make Path JSON serializable.
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved experiment results to {path}")  # Display output location.


if __name__ == "__main__":  # Run only when executed directly as a Python script.
    main()  # Start the experiment.
