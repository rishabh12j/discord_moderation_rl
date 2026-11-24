"""
Visualize user simulator dynamics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import matplotlib.pyplot as plt
import numpy as np


def visualize_warning_effects():
    """Visualize how warnings affect different user types."""
    env = DiscordEnv()
    
    # Run multiple episodes
    good_user_changes = []
    troll_changes = []
    
    for episode in range(20):
        observation, info = env.reset()
        
        for step in range(20):
            user_profile = info.get('user_profile')
            user_id = info['user_id']
            initial_tox = info['toxicity_score']
            
            # Warn everyone
            obs, reward, terminated, truncated, info = env.step(env.ACTION_WARN)
            
            # Check next message from same user
            for future_step in range(10):
                if terminated or truncated:
                    break
                
                if info.get('user_id') == user_id:
                    new_tox = info['toxicity_score']
                    change = new_tox - initial_tox
                    
                    if user_profile == 'good_user':
                        good_user_changes.append(change)
                    elif user_profile == 'troll':
                        troll_changes.append(change)
                    break
                
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(good_user_changes, bins=20, alpha=0.7, color='green')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Good Users: Toxicity Change After Warning')
    axes[0].set_xlabel('Toxicity Change')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(troll_changes, bins=20, alpha=0.7, color='red')
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2)
    axes[1].set_title('Trolls: Toxicity Change After Warning')
    axes[1].set_xlabel('Toxicity Change')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('outputs/warning_effects.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/warning_effects.png")
    plt.close()


if __name__ == "__main__":
    print("Visualizing user simulator dynamics...\n")
    visualize_warning_effects()
    print("\n✅ Visualization complete!")
