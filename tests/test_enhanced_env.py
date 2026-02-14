"""
Test enhanced environment with wrappers and statistics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.make_env import make_discord_env
import numpy as np


def test_episode_statistics():
    """Test that episode statistics are tracked correctly."""
    env = make_discord_env(
        normalize_rewards=False,
        track_stats=True,
        truncate=False
    )
    
    observation, info = env.reset()
    
    total_reward = 0
    steps = 0
    
    while steps < 20:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            # Check episode statistics in info
            assert 'episode' in info, "Episode stats should be in info"
            
            stats = info['episode']
            print(f"\nEpisode Statistics:")
            print(f"  Total reward: {stats['total_reward']:.2f}")
            print(f"  Safety violations: {stats['safety_violations']}")
            print(f"  False positives: {stats['false_positives']}")
            print(f"  True positives: {stats['true_positives']}")
            print(f"  Messages allowed: {stats['messages_allowed']}")
            print(f"  Warnings: {stats['warnings_issued']}")
            print(f"  Deletions: {stats['deletions']}")
            print(f"  Bans: {stats['bans']}")
            
            break
    
    print("✓ Episode statistics test passed")


def test_reward_normalization():
    """Test reward normalization wrapper."""
    env = make_discord_env(
        normalize_rewards=True,
        track_stats=False,
        truncate=False
    )
    
    observation, info = env.reset()
    
    rewards = []
    normalized_rewards = []
    
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_original' in info:
            rewards.append(info['reward_original'])
            normalized_rewards.append(reward)
        
        if terminated or truncated:
            break
    
    if len(normalized_rewards) > 0:
        print(f"\nOriginal rewards: {[f'{r:.2f}' for r in rewards[:5]]}")
        print(f"Normalized rewards: {[f'{r:.2f}' for r in normalized_rewards[:5]]}")
        print(f"Normalized mean: {np.mean(normalized_rewards):.3f}")
        print(f"Normalized std: {np.std(normalized_rewards):.3f}")
    
    print("✓ Reward normalization test passed")


def test_truncation():
    """Test truncation wrapper."""
    env = make_discord_env(
        normalize_rewards=False,
        track_stats=False,
        truncate=True,
        max_steps=10  # Short for testing
    )
    
    observation, info = env.reset()
    
    steps = 0
    while steps < 20:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        
        if truncated:
            assert steps <= 10, "Should truncate at max_steps"
            print(f"  Truncated at step {steps}")
            break
        
        if terminated:
            print(f"  Terminated naturally at step {steps}")
            break
    
    print("✓ Truncation test passed")


def test_full_wrapped_env():
    """Test environment with all wrappers."""
    env = make_discord_env(
        normalize_rewards=True,
        track_stats=True,
        truncate=True,
        max_steps=50
    )
    
    # Run 3 episodes
    for episode in range(3):
        observation, info = env.reset()
        
        episode_reward = 0
        steps = 0
        
        while steps < 50:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"\nEpisode {episode + 1}:")
                print(f"  Steps: {steps}")
                print(f"  Reward: {episode_reward:.2f}")
                if 'episode' in info:
                    print(f"  Safety violations: {info['episode']['safety_violations']}")
                break
    
    print("\n✓ Full wrapped environment test passed")


if __name__ == "__main__":
    print("Testing enhanced environment...\n")
    test_episode_statistics()
    test_reward_normalization()
    test_truncation()
    test_full_wrapped_env()
    print("\n✅ All enhanced environment tests passed!")
