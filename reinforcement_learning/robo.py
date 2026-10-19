import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

'''
Train a Proximal Policy Optimization (PPO) agent to balance a pole on 
a moving cart using Gymnasium and Stable-Baselines3.

The main idea is to make the agent more robust by adding random noise to its observations during training.
Stable-Baselines3 PPO's MlpPolicy uses a categorical action distribution
'''

# -----------------------------------
# Domain Randomization Wrapper
# -----------------------------------
# gym.Wrapper, a Gymnasium class that allows you to modify an existing environment without rewriting it
class NoisyCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    # Override the environment's reset() method.
    # set() initializes a new episode and returns the initial observation.
    # The **kwargs parameter allows the function to accept keyword arguments such as:env.reset(seed=42)

    def reset(self, **kwargs):
        # obs: Initial observation. info: A dictionary containing additional information.
        obs, info = self.env.reset(**kwargs)

        # Add small observation noise, size=obs.shape generates noise with the same shape as the observation
        # 0 is the mean;0.01 is the standard deviation
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
# In CartPole-v1, the default maximum episode length is 500 steps.
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