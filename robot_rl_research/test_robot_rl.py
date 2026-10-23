"""Small regression tests for the laptop robot RL prototype."""
import unittest  # Use the built-in Python testing framework.
import numpy as np  # Construct test inputs.
import torch  # Test neural-network outputs.
from robot_rl_research import PointRobot, collect_data, InverseDynamics, WorldModel, Policy


class RobotRLTests(unittest.TestCase):  # Group test cases for automated execution.
    def test_robot_action_bounds_and_sparse_reward(self):  # Verify environment safety limits.
        robot = PointRobot(horizon=2)  # Make a short episode.
        robot.reset(np.random.default_rng(1))  # Reset deterministically.
        nxt, reward, done, truncated = robot.step([99, -99])  # Provide oversized actions.
        self.assertTrue(np.isfinite(nxt).all())  # State must remain numerically valid.
        self.assertIn(reward, [0.0, 1.0])  # Reward must be sparse and binary.
        self.assertFalse(truncated)  # One step cannot truncate a two-step episode.

    def test_videos_have_no_action_labels(self):  # Verify observation-only data processing.
        data = collect_data(seed=3, episodes=6, video_episodes=3)  # Build a tiny dataset.
        self.assertEqual(len(data.videos), 3)  # Keep three observation-only trajectories.
        self.assertEqual(data.videos[0].shape[1], 4)  # Videos contain only state features.
        self.assertEqual(data.s.shape[1], 4)  # Labeled data uses four state features.
        self.assertEqual(data.a.shape[1], 2)  # Labeled actions have two control channels.

    def test_network_shapes(self):  # Verify the major model interfaces.
        s = torch.zeros(5, 4)  # Create five synthetic robot states.
        a = torch.zeros(5, 2)  # Create five synthetic robot actions.
        self.assertEqual(InverseDynamics(16)(s, s).shape, (5, 2))  # Inverse dynamics output.
        self.assertEqual(WorldModel(16)(s, a).shape, (5, 4))  # Forward dynamics output.
        self.assertEqual(Policy(16)(s).shape, (5, 2))  # Actor output.


if __name__ == "__main__":  # Allow direct execution of the test file.
    unittest.main()  # Execute all defined tests.
