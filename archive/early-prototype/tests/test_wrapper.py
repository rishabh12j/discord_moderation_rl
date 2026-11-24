import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
from src.env.wrappers import LagrangianRewardWrapper

# 1. Initialize the base environment
base_env = DiscordEnv()

# 2. Wrap it
env = LagrangianRewardWrapper(base_env, initial_lambda=0.5)

# 3. Test a step
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Base Engagement Reward: {info.get('base_reward')}")
print(f"Safety Cost Incurred: {info.get('safety_cost')}")
print(f"Final Shaped Reward (seen by PPO): {reward}")