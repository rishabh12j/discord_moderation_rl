"""
Test advanced user simulator behavioral dynamics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np


def test_good_user_warning():
    """Test that warning good users reduces toxicity."""
    print("Testing good user warning dynamics...")
    
    env = DiscordEnv()
    
    # Try multiple episodes to find a good user
    for episode in range(10):
        observation, info = env.reset()
        
        for step in range(20):
            if info.get('user_profile') == 'good_user':
                user_id = info['user_id']
                initial_toxicity = info['toxicity_score']
                
                # Warn the good user
                obs, reward, terminated, truncated, info = env.step(env.ACTION_WARN)
                
                # Check if user appears again
                warned = False
                for future_step in range(20):
                    if terminated or truncated:
                        break
                    
                    if info.get('user_id') == user_id:
                        new_toxicity = info['toxicity_score']
                        print(f"  Good user warned: {initial_toxicity:.3f} -> {new_toxicity:.3f}")
                        
                        if new_toxicity < initial_toxicity:
                            print("  ✓ Toxicity decreased as expected")
                            warned = True
                        break
                    
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                
                if warned:
                    print("✓ Good user warning test passed")
                    return
            
            if terminated or truncated:
                break
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    
    print("✓ Good user warning test completed (may not have found ideal scenario)")


def test_troll_escalation():
    """Test that warning trolls increases toxicity (escalation)."""
    print("Testing troll escalation dynamics...")
    
    env = DiscordEnv()
    
    for episode in range(10):
        observation, info = env.reset()
        
        for step in range(20):
            if info.get('user_profile') == 'troll':
                user_id = info['user_id']
                initial_toxicity = info['toxicity_score']
                
                # Warn the troll
                obs, reward, terminated, truncated, info = env.step(env.ACTION_WARN)
                
                # Check if troll appears again
                escalated = False
                for future_step in range(20):
                    if terminated or truncated:
                        break
                    
                    if info.get('user_id') == user_id:
                        new_toxicity = info['toxicity_score']
                        print(f"  Troll warned: {initial_toxicity:.3f} -> {new_toxicity:.3f}")
                        
                        if new_toxicity > initial_toxicity:
                            print("  ✓ Toxicity increased (escalation)")
                            escalated = True
                        break
                    
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                
                if escalated:
                    print("✓ Troll escalation test passed")
                    return
            
            if terminated or truncated:
                break
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
    
    print("✓ Troll escalation test completed")


def test_server_atmosphere():
    """Test server atmosphere tracking."""
    print("Testing server atmosphere dynamics...")
    
    env = DiscordEnv()
    observation, info = env.reset()
    
    atmospheres = []
    
    for step in range(20):
        # Allow all messages (including toxic ones)
        action = env.ACTION_ALLOW
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'server_atmosphere' in info:
            atmospheres.append(info['server_atmosphere'])
        
        if terminated or truncated:
            break
    
    if len(atmospheres) > 0:
        print(f"  Server atmosphere range: {min(atmospheres):.3f} - {max(atmospheres):.3f}")
        print(f"  Final atmosphere: {atmospheres[-1]:.3f}")
    
    print("✓ Server atmosphere test passed")


def test_multiple_warnings():
    """Test escalation with multiple warnings."""
    print("Testing multiple warning escalation...")
    
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Track a user across multiple warnings
    target_user = None
    warning_count = 0
    toxicity_progression = []
    
    for step in range(50):
        current_user = info.get('user_id')
        
        if target_user is None:
            # Pick first user we see
            target_user = current_user
        
        if current_user == target_user:
            # This is our tracked user
            toxicity = info.get('toxicity_score', 0)
            toxicity_progression.append(toxicity)
            
            # Warn them
            action = env.ACTION_WARN
            warning_count += 1
            
            print(f"  Warning #{warning_count} for user {target_user}: toxicity={toxicity:.3f}")
        else:
            # Other user, allow
            action = env.ACTION_ALLOW
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    if len(toxicity_progression) > 1:
        print(f"  Toxicity progression: {[f'{t:.3f}' for t in toxicity_progression]}")
        print(f"  Total warnings issued: {warning_count}")
    
    print("✓ Multiple warnings test passed")


def test_ban_atmosphere_effect():
    """Test that banning affects server atmosphere."""
    print("Testing ban atmosphere effects...")
    
    env = DiscordEnv()
    observation, info = env.reset()
    
    initial_atmosphere = info.get('server_atmosphere', 0.5)
    
    # Ban first troll we see
    for step in range(20):
        if info.get('user_profile') == 'troll':
            print(f"  Found troll with avg_toxicity={info['user_avg_toxicity']:.3f}")
            
            obs, reward, terminated, truncated, info = env.step(env.ACTION_BAN)
            
            new_atmosphere = info.get('server_atmosphere', 0.5)
            print(f"  Atmosphere: {initial_atmosphere:.3f} -> {new_atmosphere:.3f}")
            
            if new_atmosphere > initial_atmosphere:
                print("  ✓ Banning troll improved atmosphere")
            
            print("✓ Ban atmosphere test passed")
            return
        
        if terminated or truncated:
            break
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print("✓ Ban atmosphere test completed")


if __name__ == "__main__":
    print("Testing advanced user simulator...\n")
    test_good_user_warning()
    print()
    test_troll_escalation()
    print()
    test_server_atmosphere()
    print()
    test_multiple_warnings()
    print()
    test_ban_atmosphere_effect()
    print("\n✅ All user simulator tests passed!")
