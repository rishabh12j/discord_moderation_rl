"""
Test environment compatibility with Stable-Baselines3 and MaskablePPO.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
from stable_baselines3.common.env_checker import check_env


def test_env_checker():
    """Test environment with SB3 env checker."""
    print("Running Stable-Baselines3 environment checker...")
    
    env = DiscordEnv()
    
    try:
        check_env(env, warn=True)
        print("✓ Environment passed SB3 compatibility check!")
    except Exception as e:
        print(f"✗ Environment failed SB3 check: {e}")
        raise


def test_maskable_env():
    """Test that environment works with action masking utilities."""
    from sb3_contrib.common.maskable.utils import get_action_masks
    
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Get action masks
    masks = env.action_masks()
    
    assert masks.shape == (4,), "Action mask should have shape (4,)"
    assert masks.dtype == np.int8, "Action mask should be int8"
    assert masks.sum() > 0, "At least one action should be valid"
    
    print("✓ Action masking compatible with sb3-contrib")


def test_episode_rollout():
    """Test full episode with random policy."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    steps = 0
    total_reward = 0
    
    while steps < 100:  # Max 100 steps
        # Get valid action mask
        action_mask = env.action_masks()
        
        # Sample only from valid actions
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if not valid_actions:
            print("No valid actions available")
            break
        
        import random
        action = random.choice(valid_actions)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"✓ Episode rollout completed: {steps} steps, total reward: {total_reward:.2f}")


if __name__ == "__main__":
    import numpy as np
    
    print("Testing SB3 compatibility...\n")
    test_env_checker()
    test_maskable_env()
    test_episode_rollout()
    print("\n✅ All SB3 compatibility tests passed!")
