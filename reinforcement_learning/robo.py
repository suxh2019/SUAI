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
        # 0 is the mean;0.01 is the standard deviation; Implement observation-noise augmentation
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)

        return obs, info
    # The step() function is called whenever the agent takes an action.
    def step(self, action):
        #This calls the original environment's step() method. obs:Next observation,reward:Reward received,
        # terminated:Whether an MDP termination condition occurred;terminated = True means the agent reached a terminal condition, such as the pole falling beyond the allowed angle.
        # truncated = True means the episode ended because of a time limit or another external constraint. a truncated episode may still require bootstrapping a value estimate.
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add observation noise; follows Gymnasium's environment seeding mechanism
        obs = obs + self.np_random.normal(0, 0.01, size=obs.shape)

        # Small reward shaping bonus
        pole_angle = abs(obs[2])
        # This modifies the environment's original reward.The reward bonus is also affected by random observation noise.
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
    # This tells Stable-Baselines3 to use a multilayer perceptron (MLP) to represent the policy.
    # An MLP is a feedforward neural network with fully connected layers.
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    # This controls how many environment steps PPO collects per environment before performing an optimization phase.
    # With one environment, PPO collects 1,024 transitions in each rollout.n_steps=1024 is not the episode length
    # CartPole episodes may end before 1,024 steps, and PPO can collect transitions across several episodes in a single rollout.
    #n_steps=1024, After collecting the rollout, PPO splits the collected data into minibatches.
    batch_size=64,
    gamma=0.99,
)

# -----------------------------------
# Train: requests approximately 100,000 interactions with the environment.Because PPO collects 
# complete rollout batches, the actual training step count may slightly exceed 100,000.
#For this configuration, the rollout size is 1,024, so training will typically collect 99 rollouts.
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