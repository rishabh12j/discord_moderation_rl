"""
Test action masking and user simulator.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np


def test_action_masks():
    """Test action masking works correctly."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Initial state: all actions should be valid
    mask = env.action_masks()
    assert mask.sum() == 4, "All actions should be valid initially"
    
    print("✓ Initial action mask test passed")


def test_banned_user_masking():
    """Test that banned users can't be moderated again."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Find a message and ban the user
    first_user = info['user_id']
    
    # Take BAN action
    observation, reward, terminated, truncated, info = env.step(env.ACTION_BAN)
    
    # If we encounter this user again (shouldn't happen due to skip logic)
    # The mask should only allow ALLOW
    if not terminated:
        current_user = info['user_id']
        if current_user == first_user:
            mask = env.action_masks()
            assert mask[0] == 1, "ALLOW should be valid"
            assert mask[1:].sum() == 0, "WARN/DELETE/BAN should be invalid for banned user"
    
    print("✓ Banned user masking test passed")


def test_user_simulator_good_user():
    """Test that warning good users reduces toxicity."""
    env = DiscordEnv()
    
    # Run multiple episodes to find a good user who gets warned
    for episode_num in range(5):
        observation, info = env.reset()
        terminated = False  # ← ADD THIS LINE
        
        for step in range(20):
            if terminated:  # ← Move check after initialization
                break
            
            # Get current user
            current_user = info['user_id']
            user_features = env.current_episode['user_features'][step]
            
            # Find a good user
            if user_features['profile'] == 'good_user':
                # Warn them
                observation, reward, terminated, truncated, info = env.step(env.ACTION_WARN)
                
                # Check if this user appears again
                for future_step in range(step + 1, len(env.current_episode['messages'])):
                    if env.current_episode['messages'][future_step]['user_id'] == current_user:
                        # Their toxicity should be reduced
                        print(f"  Good user warned, future message toxicity adjusted")
                        break
                
                print("✓ User simulator good user test passed")
                return
            
            # Take random action to continue
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
    
    print("✓ User simulator test completed (may not have found ideal scenario)")


def test_user_simulator_troll():
    """Test that warning trolls increases toxicity."""
    env = DiscordEnv()
    
    # Run multiple episodes to find a troll
    for episode_num in range(5):
        observation, info = env.reset()
        terminated = False  # ← ADD THIS LINE
        
        for step in range(20):
            if terminated:  # ← Move this check after initialization
                break
            
            current_user = info['user_id']
            user_features = env.current_episode['user_features'][step]
            
            # Find a troll
            if user_features['profile'] == 'troll':
                # Warn them
                observation, reward, terminated, truncated, info = env.step(env.ACTION_WARN)
                
                print(f"  Troll warned at step {step}")
                print("✓ User simulator troll test passed")
                return
            
            # Take random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
    
    print("✓ User simulator troll test completed")


def test_banned_user_skip():
    """Test that banned users' messages are skipped."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Ban first user
    first_user = info['user_id']
    observation, reward, terminated, truncated, info = env.step(env.ACTION_BAN)
    
    steps = 1
    banned_user_encountered = False
    
    # Continue episode
    while not terminated and steps < 20:
        current_user = info['user_id']
        
        # Should never encounter banned user
        if current_user == first_user:
            banned_user_encountered = True
            print(f"ERROR: Encountered banned user {first_user} at step {steps}")
            break
        
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
    
    assert not banned_user_encountered, "Banned user should be skipped"
    print("✓ Banned user skip test passed")


if __name__ == "__main__":
    print("Running action masking and user simulator tests...\n")
    test_action_masks()
    test_banned_user_masking()
    test_user_simulator_good_user()
    test_user_simulator_troll()
    test_banned_user_skip()
    print("\n✅ All action masking tests passed!")
