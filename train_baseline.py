import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

env = Monitor(gym.make("HalfCheetah-v4"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("baseline_cheetah")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print("BASELINE mean reward:", mean_reward, "+/-", std_reward)

