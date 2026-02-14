"""
Comprehensive state transition tests.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np


def test_long_episode():
    """Test full episode with all features."""
    env = DiscordEnv()
    
    print("Running full episode with detailed logging...")
    
    observation, info = env.reset()
    print(f"Episode {info['conversation_id']}: {len(env.current_episode['messages'])} messages")
    
    episode_rewards = []
    safety_violations = 0
    
    step_count = 0
    while step_count < 50:  # Safety limit
        # Policy: Aggressive moderation (delete if toxic > 0.5, ban trolls)
        user_avg_tox = info['user_avg_toxicity']
        toxicity = info['toxicity_score']
        
        if toxicity > 0.8:
            action = env.ACTION_DELETE
        elif user_avg_tox > 0.6:
            action = env.ACTION_BAN
        elif toxicity > 0.3:
            action = env.ACTION_WARN
        else:
            action = env.ACTION_ALLOW
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards.append(reward)
        
        if info.get('is_toxic', False):
            safety_violations += 1
        
        print(f"Step {step_count}: {info.get('action_name', 'N/A')} | "
              f"Toxic:{info.get('is_toxic', False)}({info.get('toxicity_score', 0):.2f}) | "
              f"User:{info.get('user_profile', 'N/A')}({info.get('user_avg_toxicity', 0):.2f}) | "
              f"R:{reward:.1f}")
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode complete!")
    print(f"Total reward: {sum(episode_rewards):.1f}")
    print(f"Safety violations: {safety_violations}")
    
    # Print final stats
    if 'episode' in info:
        stats = info['episode']
        print("\nEpisode Statistics:")
        print(f"  Messages allowed: {stats['messages_allowed']}")
        print(f"  Warnings: {stats['warnings_issued']}")
        print(f"  Deletions: {stats['deletions']}")
        print(f"  Bans: {stats['bans']}")
        print(f"  True positives: {stats['true_positives']}")
        print(f"  False positives: {stats['false_positives']}")
    
    print("✓ Long episode test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    env = DiscordEnv()
    
    print("Testing edge cases...")
    
    # Test 1: Invalid action (out of bounds)
    observation, info = env.reset()
    invalid_action = 5  # Invalid! (valid range is 0-3)
    obs, rew, term, trunc, info = env.step(invalid_action)
    assert rew == -100.0, "Invalid action should give -100 reward"
    assert info['invalid_action'], "Should flag as invalid"
    print("✓ Invalid action (out of bounds) test passed")
    
    # Test 2: End of episode
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Fast-forward to end
    for _ in range(25):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    
    # Should be at end
    assert terminated, "Episode should terminate"
    assert len(env.moderation_history) > 0, "Should have history"
    print("✓ End of episode test passed")
    
    # Test 3: Banned user skipping
    env = DiscordEnv()
    observation, info = env.reset()
    first_user = info['user_id']
    
    # Ban first user
    obs, rew, term, trunc, info = env.step(env.ACTION_BAN)
    
    # Continue episode and ensure banned user never appears
    banned_encountered = False
    steps = 0
    while not term and steps < 20:
        if info.get('user_id') == first_user:
            banned_encountered = True
            print(f"ERROR: Banned user {first_user} encountered!")
            break
        
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        steps += 1
    
    assert not banned_encountered, "Banned user should never appear again"
    print(f"✓ Banned user skipping test passed (checked {steps} steps)")


def test_reward_distribution():
    """Test reward distribution across 10 episodes."""
    print("Testing reward distribution...")
    
    env = DiscordEnv()
    all_episode_rewards = []
    
    for episode_num in range(10):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 50:  # Safety limit
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode_num + 1}: reward = {episode_reward:.1f} ({steps} steps)")
    
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(all_episode_rewards):.1f}")
    print(f"  Std: {np.std(all_episode_rewards):.1f}")
    print(f"  Min: {np.min(all_episode_rewards):.1f}")
    print(f"  Max: {np.max(all_episode_rewards):.1f}")
    
    print("✓ Reward distribution test passed")


if __name__ == "__main__":
    print("Running state transition tests...\n")
    test_long_episode()
    print()
    test_edge_cases()
    print()
    test_reward_distribution()
    print("\n✅ All state transition tests passed!")
