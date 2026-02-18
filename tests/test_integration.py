"""
Integration test: Full pipeline from data to training-ready environment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.make_env import make_discord_env
import numpy as np


def test_full_pipeline():
    """Test complete pipeline with all components."""
    print("Testing full integration pipeline...\n")
    
    # Create environment with all wrappers
    env = make_discord_env(
        normalize_rewards=True,
        track_stats=True,
        truncate=True,
        max_steps=50
    )
    
    print("✓ Environment created successfully")
    
    # Run 5 episodes
    total_rewards = []
    
    for episode in range(5):
        observation, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Conversation ID: {info['conversation_id']}")
        
        episode_reward = 0
        steps = 0
        
        while steps < 50:
            # Simple heuristic policy
            toxicity = info.get('toxicity_score', 0)
            user_avg_tox = info.get('user_avg_toxicity', 0)
            
            if toxicity > 0.8:
                action = env.ACTION_DELETE
            elif user_avg_tox > 0.6:
                action = env.ACTION_BAN
            elif toxicity > 0.4:
                action = env.ACTION_WARN
            else:
                action = env.ACTION_ALLOW
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        
        if 'episode' in info:
            stats = info['episode']
            print(f"  Safety violations: {stats['safety_violations']}")
            print(f"  True positives: {stats['true_positives']}")
            print(f"  False positives: {stats['false_positives']}")
    
    print(f"\n{'='*50}")
    print("Integration Test Results:")
    print(f"  Episodes completed: {len(total_rewards)}")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    print(f"  Reward std: {np.std(total_rewards):.2f}")
    print(f"{'='*50}")
    
    print("\n✅ Full integration test passed!")


if __name__ == "__main__":
    test_full_pipeline()
