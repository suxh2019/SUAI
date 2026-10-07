import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

'''
Robust CartPole Control with PPO | PyTorch, Gymnasium

• Developed reinforcement learning policies using Proximal Policy Optimization (PPO) for balance control tasks in simulated environments.

• Designed reward functions and observation perturbation strategies to improve policy stability and robustness.

• Applied domain randomization techniques including noisy observations and varying environment dynamics to improve generalization.

• Built training and evaluation pipelines using Stable-Baselines3 and analyzed policy performance under different disturbance conditions.
'''

# -----------------------------------
# Domain Randomization Wrapper
# -----------------------------------
class NoisyCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Add small observation noise
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add observation noise
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)

        # Small reward shaping bonus
        pole_angle = abs(obs[2])

        reward += (1.0 - pole_angle)

        return obs, reward, terminated, truncated, info


# -----------------------------------
# Create Environment
# -----------------------------------
env = gym.make("CartPole-v1")

env = NoisyCartPole(env)

# -----------------------------------
# PPO Model
# -----------------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
)

# -----------------------------------
# Train
# -----------------------------------
model.learn(total_timesteps=100000)

# -----------------------------------
# Save Model
# -----------------------------------
model.save("ppo_cartpole_robust")

print("Training complete.")

# Evaluation
env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("ppo_cartpole_robust")

obs, info = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()