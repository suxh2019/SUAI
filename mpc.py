"""CPU-friendly constrained Model Predictive Control (MPC) example.

Install: pip install numpy scipy matplotlib
Run:     python mpc_laptop.py
"""

import numpy as np  # Work with state vectors, matrices, and arrays.
import matplotlib.pyplot as plt  # Plot the resulting trajectory.
from scipy.optimize import minimize  # Solve the finite-horizon optimization.
from scipy.linalg import solve_discrete_are  # Compute a stabilizing terminal cost.

# Set the discrete simulation time step (seconds).
DT = 0.1
# Define a double-integrator system: state = [position, velocity].
A = np.array([[1.0, DT], [0.0, 1.0]], dtype=float)
# Acceleration acts on both position and velocity over each time step.
B = np.array([[0.5 * DT**2], [DT]], dtype=float)
# Penalize position error more heavily than velocity error.
Q = np.diag([10.0, 1.0])
# Penalize large acceleration commands.
R = np.array([[0.1]])
# Compute a terminal cost matrix to encourage stable behavior.
P = solve_discrete_are(A, B, Q, R)
# Number of future control actions optimized at each time step.
HORIZON = 20
# Restrict commanded acceleration to physically plausible values.
MAX_ACCEL = 2.0
# Number of simulated control steps.
SIM_STEPS = 100
# Begin 5 meters away from the target with zero velocity.
INITIAL_STATE = np.array([5.0, 0.0])
# Desired equilibrium is zero position and zero velocity.
TARGET_STATE = np.array([0.0, 0.0])


def predict_trajectory(initial_state, controls):
    """Simulate the model over the planning horizon."""
    state = initial_state.copy()  # Avoid altering the caller's state.
    states = [state.copy()]  # Include the initial state in predictions.
    for u in controls:  # Process the candidate accelerations in sequence.
        state = A @ state + B[:, 0] * u  # Apply linear dynamics.
        states.append(state.copy())  # Save the next predicted state.
    return np.asarray(states)  # Return an array of shape [horizon + 1, 2].


def mpc_objective(controls, initial_state):
    """Compute the quadratic predicted trajectory cost."""
    states = predict_trajectory(initial_state, controls)  # Predict states.
    cost = 0.0  # Accumulate running and terminal costs.
    for k, u in enumerate(controls):  # Score each predicted control step.
        error = states[k] - TARGET_STATE  # Measure tracking error.
        cost += error @ Q @ error  # Penalize position and velocity error.
        cost += R[0, 0] * u**2  # Penalize large actions.
    final_error = states[-1] - TARGET_STATE  # Measure terminal error.
    cost += final_error @ P @ final_error  # Add terminal penalty.
    return float(cost)  # SciPy expects a scalar floating-point objective.


def solve_mpc(state, initial_guess):
    """Optimize the control sequence and return its first action."""
    bounds = [(-MAX_ACCEL, MAX_ACCEL)] * HORIZON  # Bound each action.
    result = minimize(  # Run a constrained nonlinear optimizer.
        mpc_objective,  # The function to minimize.
        x0=initial_guess,  # Warm-start with the previous solution.
        args=(state,),  # Pass the current measured state to the cost.
        method="SLSQP",  # Supports the input bounds specified above.
        bounds=bounds,  # Apply acceleration constraints.
        options={"maxiter": 150, "ftol": 1e-7, "disp": False},  # Accuracy settings.
    )
    if not result.success:  # Explicitly handle solver failure.
        raise RuntimeError(f"MPC optimization failed: {result.message}")
    return float(result.x[0]), result.x  # Apply only the FIRST planned action.


def main():
    state = INITIAL_STATE.copy()  # Initialize the true simulated system.
    guess = np.zeros(HORIZON)  # Initial candidate control sequence.
    state_log = [state.copy()]  # Record initial position and velocity.
    action_log = []  # Record the applied accelerations.

    for step in range(SIM_STEPS):  # Repeat the receding-horizon MPC loop.
        action, plan = solve_mpc(state, guess)  # Re-optimize from current state.
        state = A @ state + B[:, 0] * action  # Apply one action to the system.
        state_log.append(state.copy())  # Save the resulting state.
        action_log.append(action)  # Save the applied input.
        guess = np.r_[plan[1:], plan[-1]]  # Shift the plan for warm-starting.
        if (step + 1) % 20 == 0:  # Report progress every 20 steps.
            print(f"Step {step + 1:3d}: position={state[0]: .4f}, "
                  f"velocity={state[1]: .4f}, action={action: .4f}")

    state_log = np.asarray(state_log)  # Convert states to an array.
    action_log = np.asarray(action_log)  # Convert actions to an array.
    print("Final state:", state_log[-1])  # Report tracking performance.
    print("Maximum absolute acceleration:", np.max(np.abs(action_log)))

    times = np.arange(SIM_STEPS + 1) * DT  # Times for state measurements.
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)  # Create plots.
    axes[0].plot(times, state_log[:, 0])  # Plot position over time.
    axes[0].axhline(0, linestyle="--", alpha=0.5)  # Mark desired position.
    axes[0].set_ylabel("Position (m)")  # Label the position axis.
    axes[1].plot(times, state_log[:, 1])  # Plot velocity over time.
    axes[1].axhline(0, linestyle="--", alpha=0.5)  # Mark desired velocity.
    axes[1].set_ylabel("Velocity (m/s)")  # Label the velocity axis.
    axes[2].step(times[:-1], action_log, where="post")  # Plot acceleration.
    axes[2].axhline(MAX_ACCEL, linestyle="--", alpha=0.5)  # Upper bound.
    axes[2].axhline(-MAX_ACCEL, linestyle="--", alpha=0.5)  # Lower bound.
    axes[2].set_ylabel("Acceleration (m/s²)")  # Label the control axis.
    axes[2].set_xlabel("Time (seconds)")  # Label the horizontal axis.
    fig.suptitle("Constrained Model Predictive Control (MPC)")  # Add title.
    fig.tight_layout()  # Prevent labels from overlapping.
    fig.savefig("mpc_results.png", dpi=160)  # Save the figure to a file.
    plt.show()  # Display the figure when running locally.


if __name__ == "__main__":  # Only run simulation when executed directly.
    main()  # Start the MPC experiment.
