"""
Analyze reward signal distribution and properties.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np
import matplotlib.pyplot as plt


def analyze_reward_distribution():
    """Analyze reward distribution across episodes."""
    env = DiscordEnv()
    
    all_step_rewards = []
    reward_by_action = {0: [], 1: [], 2: [], 3: []}
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("Analyzing reward signals across 100 episodes...")
    
    for episode in range(100):
        observation, info = env.reset()
        
        for step in range(50):
            # Random action
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            all_step_rewards.append(reward)
            reward_by_action[action].append(reward)
            action_counts[action] += 1
            
            if terminated or truncated:
                break
    
    # Analysis
    print("\n" + "="*60)
    print("Reward Signal Analysis")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"  Mean: {np.mean(all_step_rewards):.3f}")
    print(f"  Std:  {np.std(all_step_rewards):.3f}")
    print(f"  Min:  {np.min(all_step_rewards):.3f}")
    print(f"  Max:  {np.max(all_step_rewards):.3f}")
    print(f"  Median: {np.median(all_step_rewards):.3f}")
    
    print(f"\nReward by Action:")
    action_names = ['ALLOW', 'WARN', 'DELETE', 'BAN']
    for action_id, name in enumerate(action_names):
        if len(reward_by_action[action_id]) > 0:
            mean_reward = np.mean(reward_by_action[action_id])
            count = action_counts[action_id]
            print(f"  {name:<8}: mean={mean_reward:>7.2f}, count={count}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    axes[0, 0].hist(all_step_rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Overall Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    
    # By action
    for action_id, name in enumerate(action_names):
        row = (action_id // 2) if action_id > 0 else 0
        col = (action_id % 2) if action_id > 0 else 1
        
        if action_id == 0:
            ax = axes[0, 1]
        else:
            ax = axes[row, col - 1] if col > 0 else axes[row, 1]
        
        if len(reward_by_action[action_id]) > 0:
            ax.hist(reward_by_action[action_id], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{name} Action Rewards')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('outputs/reward_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to outputs/reward_analysis.png")
    plt.close()
    
    # Check for sparse rewards
    zero_rewards = np.sum(np.array(all_step_rewards) == 0)
    print(f"\nReward Sparsity:")
    print(f"  Zero rewards: {zero_rewards}/{len(all_step_rewards)} ({zero_rewards/len(all_step_rewards)*100:.1f}%)")
    
    # Check for extreme outliers
    q1, q3 = np.percentile(all_step_rewards, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    outliers = [r for r in all_step_rewards if r < lower_bound or r > upper_bound]
    print(f"  Extreme outliers: {len(outliers)}/{len(all_step_rewards)} ({len(outliers)/len(all_step_rewards)*100:.1f}%)")
    
    print("\n✓ Reward analysis complete")


if __name__ == "__main__":
    analyze_reward_distribution()
