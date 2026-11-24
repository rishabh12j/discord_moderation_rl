"""
Interactive demo of Discord moderation environment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import time


def demo_episode():
    """Run interactive demo of one episode."""
    env = DiscordEnv()
    
    print("="*60)
    print("Discord Moderation RL Environment - Demo")
    print("="*60)
    print("\nActions:")
    print("  0 = ALLOW  (let message through)")
    print("  1 = WARN   (warn the user)")
    print("  2 = DELETE (remove message)")
    print("  3 = BAN    (ban user from server)")
    print("  q = quit")
    print("="*60)
    
    observation, info = env.reset()
    
    print(f"\nStarting Episode: Conversation {info['conversation_id']}")
    print(f"Total messages: {len(env.current_episode['messages'])}\n")
    
    episode_reward = 0
    
    while True:
        # Display current message
        print("\n" + "="*60)
        print(f"Step {info['step'] + 1}/{len(env.current_episode['messages'])}")
        print("="*60)
        
        print(f"\n📝 Message: \"{info['message_text']}\"")
        print(f"\n👤 User Profile:")
        print(f"   - ID: {info['user_id']}")
        print(f"   - Type: {info['user_profile']}")
        print(f"   - Avg Toxicity: {info['user_avg_toxicity']:.3f}")
        print(f"   - Warnings: {info['warnings_this_user']}")
        
        print(f"\n🔍 Message Analysis:")
        print(f"   - Toxicity Score: {info['toxicity_score']:.3f}")
        print(f"   - Is Toxic: {'⚠️  YES' if info['is_toxic'] else '✅ NO'}")
        
        print(f"\n🏢 Server Stats:")
        print(f"   - Atmosphere: {info['server_atmosphere']:.3f}")
        print(f"   - Avg Toxicity: {info['server_avg_toxicity']:.3f}")
        print(f"   - Banned Users: {info['banned_users_count']}")
        
        # Get action from user
        valid_actions = info['valid_actions']
        action_names = ['ALLOW', 'WARN', 'DELETE', 'BAN']
        
        print(f"\n🎮 Valid Actions: {[action_names[i] for i, v in enumerate(valid_actions) if v]}")
        
        while True:
            user_input = input("\nYour action (0-3 or q): ").strip().lower()
            
            if user_input == 'q':
                print("\nExiting demo...")
                return
            
            try:
                action = int(user_input)
                if 0 <= action <= 3 and valid_actions[action]:
                    break
                else:
                    print(f"❌ Invalid action. Choose from: {[i for i, v in enumerate(valid_actions) if v]}")
            except ValueError:
                print("❌ Please enter a number 0-3 or 'q'")
        
        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        
        print(f"\n💰 Reward: {reward:+.1f} (Total: {episode_reward:+.1f})")
        
        if terminated or truncated:
            print("\n" + "="*60)
            print("Episode Complete!")
            print("="*60)
            
            if 'episode' in info:
                stats = info['episode']
                print(f"\n📊 Final Statistics:")
                print(f"   - Total Reward: {stats['total_reward']:.1f}")
                print(f"   - Messages Allowed: {stats['messages_allowed']}")
                print(f"   - Safety Violations: {stats['safety_violations']}")
                print(f"   - True Positives: {stats['true_positives']}")
                print(f"   - False Positives: {stats['false_positives']}")
                print(f"   - Warnings: {stats['warnings_issued']}")
                print(f"   - Deletions: {stats['deletions']}")
                print(f"   - Bans: {stats['bans']}")
            
            break
        
        time.sleep(0.5)  # Brief pause for readability


if __name__ == "__main__":
    try:
        demo_episode()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
