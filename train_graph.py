import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from graph_model import GraphFeaturesExtractor

env = Monitor(gym.make("HalfCheetah-v4"))

# tell PPO to use our graph extractor instead of the default blob
policy_kwargs = dict(
    features_extractor_class=GraphFeaturesExtractor,
    features_extractor_kwargs=dict(hidden_dim=32),
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# full training run: same 1 million steps as the baseline
model.learn(total_timesteps=1000000)
model.save("graph_cheetah")

# evaluate: 10 episodes, averaged
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print("GRAPH mean reward:", mean_reward, "+/-", std_reward)

