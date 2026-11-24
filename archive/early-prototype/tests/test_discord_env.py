"""
Test Discord environment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np


def test_initialization():
    """Test environment initializes correctly."""
    env = DiscordEnv()
    
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.n == 4
    
    print("✓ Initialization test passed")


def test_reset():
    """Test reset returns valid observation."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    # Check observation structure
    assert 'message_embedding' in observation
    assert 'user_history' in observation
    assert 'server_stats' in observation
    
    # Check shapes
    assert observation['message_embedding'].shape == (384,)
    assert observation['user_history'].shape == (5,)
    assert observation['server_stats'].shape == (3,)
    
    # Check info
    assert 'conversation_id' in info
    assert 'step' in info
    
    print("✓ Reset test passed")


def test_step():
    """Test step returns valid outputs."""
    env = DiscordEnv()
    env.reset()
    
    observation, reward, terminated, truncated, info = env.step(0)
    
    # Check outputs
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    print("✓ Step test passed")


def test_episode_completion():
    """Test full episode runs without errors."""
    env = DiscordEnv()
    observation, info = env.reset()
    
    steps = 0
    total_reward = 0
    
    while steps < 20:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    assert steps > 0
    print(f"✓ Episode completion test passed ({steps} steps, reward: {total_reward:.2f})")


def test_observation_space_sample():
    """Test observation space sampling."""
    env = DiscordEnv()
    
    # Gymnasium should be able to sample from observation space
    sample = env.observation_space.sample()
    
    assert env.observation_space.contains(sample)
    
    print("✓ Observation space sample test passed")


if __name__ == "__main__":
    print("Running Discord environment tests...\n")
    test_initialization()
    test_reset()
    test_step()
    test_episode_completion()
    test_observation_space_sample()
    print("\n✅ All environment tests passed!")
