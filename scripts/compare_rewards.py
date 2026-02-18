"""
Compare different reward configurations side-by-side.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
from src.env.reward_configs import BASELINE, BALANCED, SAFETY_FIRST, ENGAGEMENT_FIRST
from src.utils.episode_builder import EpisodeBuilder
from src.utils.toxicity_judge import ToxicityJudge
import numpy as np
import matplotlib.pyplot as plt


def compare_configurations():
    """Compare predefined reward configurations."""
    
    configs = [
        ("Baseline", BASELINE),
        ("Balanced", BALANCED),
        ("Safety First", SAFETY_FIRST),
        ("Engagement First", ENGAGEMENT_FIRST)
    ]
    
    # Share components across all environments for efficiency
    print("Initializing shared components...")
    episode_builder = EpisodeBuilder()
    toxicity_judge = ToxicityJudge(device='cuda', batch_size=32)
    
    results = {}
    
    print("\nComparing reward configurations...\n")
    
    for name, weights in configs:
        print(f"Testing {name}...")
        print(f"  Engagement reward: {weights.engagement_reward}")
        print(f"  Safety penalty: {weights.safety_penalty}")
        
        # Create environment with specific reward weights
        env = DiscordEnv(
            episode_builder=episode_builder,
            toxicity_judge=toxicity_judge,
            reward_weights=weights
        )
        
        rewards = []
        safety_viols = []
        false_positives = []
        messages_allowed = []
        
        for episode in range(20):
            obs, info = env.reset()
            ep_reward = 0
            steps = 0
            
            while steps < 50:
                # Simple heuristic policy
                tox = info['toxicity_score']
                user_avg = info['user_avg_toxicity']
                
                # Heuristic moderation strategy
                if user_avg > 0.6:
                    action = env.ACTION_BAN
                elif tox > 0.7:
                    action = env.ACTION_DELETE
                elif tox > 0.4:
                    action = env.ACTION_WARN
                else:
                    action = env.ACTION_ALLOW
                
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += reward
                steps += 1
                
                if term or trunc:
                    break
            
            rewards.append(ep_reward)
            if 'episode' in info:
                stats = info['episode']
                safety_viols.append(stats['safety_violations'])
                false_positives.append(stats['false_positives'])
                messages_allowed.append(stats['messages_allowed'])
        
        results[name] = {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'safety_mean': np.mean(safety_viols),
            'false_positive_mean': np.mean(false_positives),
            'messages_allowed_mean': np.mean(messages_allowed)
        }
        
        print(f"  Results:")
        print(f"    Reward: {results[name]['reward_mean']:.1f} ± {results[name]['reward_std']:.1f}")
        print(f"    Safety violations: {results[name]['safety_mean']:.2f}")
        print(f"    False positives: {results[name]['false_positive_mean']:.2f}")
        print(f"    Messages allowed: {results[name]['messages_allowed_mean']:.1f}\n")
    
    # Visualize
    print("Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    rewards_list = [results[n]['reward_mean'] for n in names]
    safety = [results[n]['safety_mean'] for n in names]
    false_pos = [results[n]['false_positive_mean'] for n in names]
    messages = [results[n]['messages_allowed_mean'] for n in names]
    
    # Plot 1: Average reward
    axes[0, 0].barh(names, rewards_list, color='steelblue')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Average Episode Reward')
    axes[0, 0].set_title('Reward by Configuration')
    
    # Plot 2: Safety violations
    axes[0, 1].barh(names, safety, color='coral')
    axes[0, 1].set_xlabel('Avg Safety Violations per Episode')
    axes[0, 1].set_title('Safety Performance')
    
    # Plot 3: False positives
    axes[1, 0].barh(names, false_pos, color='orange')
    axes[1, 0].set_xlabel('Avg False Positives per Episode')
    axes[1, 0].set_title('Fairness (Lower is Better)')
    
    # Plot 4: Messages allowed (engagement)
    axes[1, 1].barh(names, messages, color='green')
    axes[1, 1].set_xlabel('Avg Messages Allowed per Episode')
    axes[1, 1].set_title('Engagement (Higher is Better)')
    
    plt.tight_layout()
    
    output_path = Path('outputs/config_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary Comparison Table")
    print("="*80)
    print(f"{'Config':<20} {'Reward':<15} {'Safety':<12} {'False Pos':<12} {'Engagement':<12}")
    print("-"*80)
    for name in names:
        r = results[name]
        print(f"{name:<20} {r['reward_mean']:>7.1f} ± {r['reward_std']:<4.1f} "
              f"{r['safety_mean']:>6.2f}      {r['false_positive_mean']:>6.2f}      "
              f"{r['messages_allowed_mean']:>6.1f}")
    print("="*80)


if __name__ == "__main__":
    np.random.seed(42)
    compare_configurations()
    print("\n✅ Configuration comparison complete!")
