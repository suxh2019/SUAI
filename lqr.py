"""Discrete-time Linear Quadratic Regulator (LQR), CPU-friendly example."""

import numpy as np  # Use NumPy for vectors, matrices, and matrix multiplication.
import matplotlib.pyplot as plt  # Plot the simulated states and control actions.
from scipy.linalg import solve_discrete_are  # Solve the discrete algebraic Riccati equation.


def compute_lqr_gain(A, B, Q, R):  # Compute the optimal infinite-horizon LQR feedback gain.
    P = solve_discrete_are(A, B, Q, R)  # Solve P = A'PA - A'PB(R+B'PB)^-1 B'PA + Q.
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)  # Calculate K without explicit matrix inversion.
    return K, P  # Return the feedback matrix and Riccati solution.


def main():  # Run the complete demonstration.
    dt = 0.1  # Set the simulation sampling interval to 0.1 seconds.
    A = np.array([[1.0, dt], [0.0, 1.0]])  # Model position/velocity for a double integrator.
    B = np.array([[0.5 * dt**2], [dt]])  # Model how acceleration changes position and velocity.
    Q = np.diag([10.0, 1.0])  # Penalize position error more strongly than velocity.
    R = np.array([[0.1]])  # Penalize large control accelerations.
    K, P = compute_lqr_gain(A, B, Q, R)  # Solve for the optimal linear feedback controller.
    print("LQR gain K =", K)  # Print the feedback gains for position and velocity.
    print("Closed-loop eigenvalues =", np.linalg.eigvals(A - B @ K))  # Verify stability: magnitudes < 1.

    steps = 100  # Simulate 100 time steps (10 seconds).
    x = np.array([5.0, 0.0])  # Start at position 5 meters and velocity 0 meters/second.
    states = [x.copy()]  # Keep the initial state for plotting.
    controls = []  # Store control actions for plotting.

    for _ in range(steps):  # Advance the closed-loop system one time step at a time.
        u = float((-K @ x).item())  # Apply the optimal feedback rule u = -Kx.
        x = A @ x + B[:, 0] * u  # Compute the next state with x_next = Ax + Bu.
        controls.append(u)  # Record the applied acceleration.
        states.append(x.copy())  # Record a copy of the updated state.

    states = np.asarray(states)  # Convert the state trajectory into a NumPy array.
    controls = np.asarray(controls)  # Convert the action history into a NumPy array.
    state_times = np.arange(steps + 1) * dt  # Compute timestamps for the stored states.
    control_times = np.arange(steps) * dt  # Compute timestamps for the stored controls.

    print("Final state [position, velocity] =", states[-1])  # Show how close the system gets to zero.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)  # Create state and control plots.
    axes[0].plot(state_times, states[:, 0], label="Position (m)")  # Plot position over time.
    axes[0].plot(state_times, states[:, 1], label="Velocity (m/s)")  # Plot velocity over time.
    axes[0].axhline(0, color="gray", linewidth=0.8)  # Mark the zero target state.
    axes[0].set_ylabel("State")  # Label the vertical axis for the state plot.
    axes[0].legend()  # Identify the position and velocity curves.
    axes[0].grid(True)  # Add a grid for easier reading.
    axes[1].step(control_times, controls, where="post", label="Acceleration (m/s²)")  # Plot control actions.
    axes[1].set_xlabel("Time (seconds)")  # Label the horizontal time axis.
    axes[1].set_ylabel("Control")  # Label the control axis.
    axes[1].legend()  # Show the control legend.
    axes[1].grid(True)  # Add a grid for easier reading.
    fig.suptitle("Discrete-Time LQR: Double Integrator")  # Give the whole figure a title.
    fig.tight_layout()  # Prevent labels from overlapping.
    fig.savefig("lqr_results.png", dpi=150)  # Save the plot even on laptops without a display.
    print("Saved figure to lqr_results.png")  # Confirm where the plot was saved.
    plt.show()  # Open an interactive plot window when the environment supports it.


if __name__ == "__main__":  # Only execute the demo when this file is run directly.
    main()  # Start the simulation.
