"""
Demonstrate user simulator dynamics with visualizations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv


def demo_user_reactions():
    """Demonstrate how users react to moderation."""
    print("=" * 70)
    print("User Simulator Dynamics Demonstration")
    print("=" * 70)
    
    env = DiscordEnv()
    
    # Demo 1: Warning a good user
    print("\n" + "=" * 70)
    print("Scenario 1: Warning a Good User")
    print("=" * 70)
    
    for attempt in range(10):
        observation, info = env.reset()
        
        # Find a good user
        for step in range(20):
            user_features = env.current_episode['user_features'][step]
            
            if user_features['profile'] == 'good_user':
                current_message = env.current_episode['messages'][step]
                print(f"\nFound good user: {current_message['user_id']}")
                print(f"Current message toxicity: {current_message['toxicity_score']:.3f}")
                
                # Warn them
                env.current_step = step
                env.step(env.ACTION_WARN)
                
                # Check future messages
                print("\nFuture messages from this user:")
                for future_step in range(step + 1, min(step + 4, len(env.current_episode['messages']))):
                    future_msg = env.current_episode['messages'][future_step]
                    if future_msg['user_id'] == current_message['user_id']:
                        print(f"  Step {future_step}: toxicity = {future_msg['toxicity_score']:.3f} "
                              f"(reduced by ~30%)")
                
                break
            
            if step == 19:
                continue  # Try next episode
        else:
            continue
        break
    
    # Demo 2: Banning a troll
    print("\n" + "=" * 70)
    print("Scenario 2: Banning a Troll")
    print("=" * 70)
    
    for attempt in range(10):
        observation, info = env.reset()
        
        for step in range(20):
            user_features = env.current_episode['user_features'][step]
            
            if user_features['profile'] == 'troll':
                current_message = env.current_episode['messages'][step]
                troll_id = current_message['user_id']
                print(f"\nFound troll: {troll_id}")
                print(f"Troll's avg toxicity: {user_features['avg_toxicity']:.3f}")
                print(f"Current message: {current_message['text'][:60]}...")
                
                # Ban them
                env.current_step = step
                observation, reward, terminated, truncated, info = env.step(env.ACTION_BAN)
                
                print(f"\nBanned troll at step {step}")
                print(f"Reward for banning: {reward:.2f}")
                print(f"Troll now in banned list: {troll_id in env.banned_users}")
                
                # Check that future messages skip this user
                print("\nNext 3 messages (troll's messages should be skipped):")
                for i in range(3):
                    if not terminated:
                        next_user = info['user_id']
                        print(f"  Step {env.current_step}: User {next_user} "
                              f"{'← SKIPPED TROLL' if next_user != troll_id else ''}")
                        
                        action = env.action_space.sample()
                        observation, reward, terminated, truncated, info = env.step(action)
                
                break
        else:
            continue
        break
    
    print("\n" + "=" * 70)
    print("✅ User Simulator Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_user_reactions()
