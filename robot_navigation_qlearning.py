"""CPU-friendly robotics / embodied-AI / Q-learning demonstration.

Run: python robot_navigation_qlearning.py
Uses only the Python standard library.
"""

import random  # Generate random starting states and exploration actions.
from collections import defaultdict  # Create new Q-value rows automatically.

# Specify grid dimensions; (row, column) identifies a robot position.
HEIGHT = 7  # Number of rows in the simulated environment.
WIDTH = 7  # Number of columns in the simulated environment.
GOAL = (0, 6)  # The robot's target location.
OBSTACLES = {  # Cells occupied by impassable obstacles.
    (1, 2), (2, 2), (3, 2), (4, 2),  # A vertical wall.
    (2, 4), (3, 4), (4, 4),  # A second wall.
    (5, 5),  # Another obstacle near the bottom.
}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right.
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]  # Human-readable action names.
MAX_STEPS = 60  # Safety limit: maximum actions per training episode.


class RobotWorld:
    """A minimal environment with robot state, movement, and rewards."""

    def __init__(self):  # Initialize the simulated world.
        self.position = (6, 0)  # Place the robot at an initial location.

    def reset(self):  # Start a new episode with a random valid position.
        free = [  # Collect possible non-goal, non-obstacle starting cells.
            (r, c) for r in range(HEIGHT) for c in range(WIDTH)
            if (r, c) not in OBSTACLES and (r, c) != GOAL
        ]
        self.position = random.choice(free)  # Randomize the starting position.
        return self.position  # Return the robot's observed state.

    def step(self, action):  # Apply an action to the simulated robot.
        dr, dc = ACTIONS[action]  # Get row/column movement for the action.
        r, c = self.position  # Read the current robot position.
        candidate = (r + dr, c + dc)  # Compute the desired next position.
        valid = (  # Check map boundaries and obstacle collisions.
            0 <= candidate[0] < HEIGHT
            and 0 <= candidate[1] < WIDTH
            and candidate not in OBSTACLES
        )
        if not valid:  # If movement is blocked by a wall or boundary...
            return self.position, -1.0, False  # Stay in place and penalize collision.
        self.position = candidate  # Move the robot to the valid new location.
        if self.position == GOAL:  # Check whether the target was reached.
            return self.position, 10.0, True  # Reward success and end the episode.
        return self.position, -0.1, False  # Charge a small cost for each movement.

    def render(self):  # Print the robot and environment as an ASCII grid.
        for r in range(HEIGHT):  # Visit each row of the grid.
            symbols = []  # Store a printable character for every cell.
            for c in range(WIDTH):  # Visit each column of the grid.
                cell = (r, c)  # Construct this cell's coordinates.
                if cell == self.position:  # Robot has the highest visual priority.
                    symbol = "R"  # R represents the robot.
                elif cell == GOAL:  # Mark the destination.
                    symbol = "G"  # G represents the goal.
                elif cell in OBSTACLES:  # Mark blocked locations.
                    symbol = "#"  # # represents an obstacle.
                else:  # Otherwise the robot can occupy the cell.
                    symbol = "."  # Dot represents empty space.
                symbols.append(symbol)  # Add this cell to the current row.
            print(" ".join(symbols))  # Print the completed row.
        print()  # Add a blank line for readability.


def choose_action(q_table, state, epsilon):  # Epsilon-greedy robot policy.
    if random.random() < epsilon:  # With probability epsilon, explore.
        return random.randrange(4)  # Choose one of four actions randomly.
    values = q_table[state]  # Otherwise examine learned action values.
    best_value = max(values)  # Identify the highest expected action value.
    best_actions = [i for i, v in enumerate(values) if v == best_value]  # Handle ties.
    return random.choice(best_actions)  # Exploit a randomly selected best action.


def train(episodes=2500):  # Learn action values using tabular Q-learning.
    env = RobotWorld()  # Create the simulated environment.
    q_table = defaultdict(lambda: [0.0] * 4)  # Four Q-values per state.
    alpha = 0.20  # Learning rate: how strongly to incorporate new information.
    gamma = 0.95  # Discount factor for future rewards.
    success_count = 0  # Track recent successes for progress reporting.
    for episode in range(episodes):  # Repeat the robot's trial-and-error experience.
        state = env.reset()  # Begin a new episode at a random position.
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.7))  # Gradually explore less.
        for _ in range(MAX_STEPS):  # Limit the length of this episode.
            action = choose_action(q_table, state, epsilon)  # Select robot action.
            next_state, reward, done = env.step(action)  # Interact with the world.
            old_value = q_table[state][action]  # Read the old estimate Q(s,a).
            future_value = 0.0 if done else max(q_table[next_state])  # Bootstrap.
            target = reward + gamma * future_value  # Build Bellman target.
            q_table[state][action] += alpha * (target - old_value)  # Q-learning update.
            state = next_state  # Advance to the next robot observation.
            if done:  # Stop once the goal is reached.
                success_count += 1  # Record a successful training episode.
                break  # Exit this episode's action loop.
        if (episode + 1) % 500 == 0:  # Print progress after each 500 episodes.
            print(f"Episode {episode + 1:4d} | recent successes: {success_count}/500")
            success_count = 0  # Reset the reporting counter.
    return q_table  # Return the learned state-action value table.


def evaluate(q_table, trials=100):  # Check performance without random exploration.
    env = RobotWorld()  # Create a fresh robot environment.
    successes = 0  # Count goals reached by the learned policy.
    for _ in range(trials):  # Test on several randomly sampled starting cells.
        state = env.reset()  # Start a new evaluation episode.
        for _ in range(MAX_STEPS):  # Follow the policy up to the time limit.
            action = choose_action(q_table, state, epsilon=0.0)  # Greedy action.
            state, reward, done = env.step(action)  # Execute the action.
            if done:  # Stop the rollout when the robot reaches its destination.
                successes += 1  # Count this test as a success.
                break  # Move on to another start position.
    print(f"Evaluation success rate: {successes}/{trials} = {successes / trials:.1%}")


def demonstrate(q_table):  # Show a learned path from a fixed starting cell.
    env = RobotWorld()  # Create a robot simulation.
    env.position = (6, 0)  # Choose a reproducible starting point.
    print("\nInitial environment (R=robot, G=goal, #=wall):")  # Explain map symbols.
    env.render()  # Display the grid before the robot moves.
    state = env.position  # Read the robot's initial state.
    path = [state]  # Record the sequence of visited positions.
    for step in range(MAX_STEPS):  # Run the trained policy.
        action = choose_action(q_table, state, epsilon=0.0)  # Choose best action.
        state, reward, done = env.step(action)  # Move the robot.
        path.append(state)  # Save its new position.
        print(f"Step {step + 1:2d}: {ACTION_NAMES[action]:5s} -> {state}, reward={reward:+.1f}")
        if done:  # Stop the demonstration if the target was reached.
            print("Success: robot reached the goal!")  # Report success.
            break  # Exit the rollout loop.
    else:  # This executes if the loop finishes without reaching the goal.
        print("Robot did not reach the goal within the step limit.")
    print("\nFinal environment:")  # Label the final map.
    env.render()  # Print the robot's final location.
    print("Visited path:", path)  # Show the complete path for inspection.


if __name__ == "__main__":  # Only execute training when run as a script.
    random.seed(42)  # Make the demonstration mostly reproducible.
    learned_q = train()  # Learn navigation behavior from experience.
    evaluate(learned_q)  # Quantify policy success after learning.
    demonstrate(learned_q)  # Print an example autonomous navigation trajectory.
