"""
Profile environment performance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import time
import numpy as np


def profile_environment():
    """Profile environment reset and step performance."""
    env = DiscordEnv()
    
    print("Profiling environment performance...\n")
    
    # Profile reset
    reset_times = []
    for _ in range(100):
        start = time.time()
        env.reset()
        reset_times.append(time.time() - start)
    
    print(f"Reset Performance:")
    print(f"  Mean: {np.mean(reset_times)*1000:.2f}ms")
    print(f"  Std:  {np.std(reset_times)*1000:.2f}ms")
    print(f"  Min:  {np.min(reset_times)*1000:.2f}ms")
    print(f"  Max:  {np.max(reset_times)*1000:.2f}ms")
    
    # Profile step
    step_times = []
    env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        
        start = time.time()
        observation, reward, terminated, truncated, info = env.step(action)
        step_times.append(time.time() - start)
        
        if terminated or truncated:
            env.reset()
    
    print(f"\nStep Performance:")
    print(f"  Mean: {np.mean(step_times)*1000:.2f}ms")
    print(f"  Std:  {np.std(step_times)*1000:.2f}ms")
    print(f"  Min:  {np.min(step_times)*1000:.2f}ms")
    print(f"  Max:  {np.max(step_times)*1000:.2f}ms")
    
    # Calculate throughput
    fps = 1.0 / np.mean(step_times)
    print(f"\nThroughput: {fps:.0f} steps/second")
    
    # Estimate training time
    total_timesteps = 1_000_000
    estimated_hours = (total_timesteps / fps) / 3600
    print(f"\nEstimated time for 1M timesteps: {estimated_hours:.1f} hours")
    
    print("\n✓ Profiling complete")


if __name__ == "__main__":
    profile_environment()
