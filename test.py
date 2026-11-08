import gymnasium as gym
env = gym.make("HalfCheetah-v4")
obs, info = env.reset()
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("It works. Observation size:", obs.shape)

