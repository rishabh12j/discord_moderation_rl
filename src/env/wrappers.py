"""
Gymnasium wrappers for Discord environment.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any


class RewardNormalizationWrapper(gym.Wrapper):
    """
    Normalize rewards to improve learning stability.
    
    Tracks running mean and std of rewards and normalizes:
    normalized_reward = (reward - mean) / (std + epsilon)
    """
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, clip_range: float = 10.0):
        """
        Initialize reward normalization wrapper.
        
        Args:
            env: Environment to wrap
            epsilon: Small value to avoid division by zero
            clip_range: Clip normalized rewards to [-clip_range, clip_range]
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Running statistics
        self.reward_sum = 0.0
        self.reward_sum_sq = 0.0
        self.reward_count = 0
        
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with reward normalization."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update running statistics
        self.reward_sum += reward
        self.reward_sum_sq += reward ** 2
        self.reward_count += 1
        
        # Calculate running mean and std
        mean = self.reward_sum / self.reward_count
        variance = (self.reward_sum_sq / self.reward_count) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        # Normalize reward
        normalized_reward = (reward - mean) / (std + self.epsilon)
        
        # Clip to prevent extreme values
        normalized_reward = np.clip(normalized_reward, -self.clip_range, self.clip_range)
        
        # Store original reward in info
        info['reward_original'] = reward
        info['reward_mean'] = mean
        info['reward_std'] = std
        
        return observation, float(normalized_reward), terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset environment."""
        return self.env.reset(**kwargs)


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Track and log episode-level statistics.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize episode stats wrapper."""
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with episode tracking."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if terminated or truncated:
            # Episode ended - log statistics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Add to info
            info['episode_reward'] = self.current_episode_reward
            info['episode_length'] = self.current_episode_length
            
            if len(self.episode_rewards) >= 100:
                # Calculate rolling statistics
                recent_rewards = self.episode_rewards[-100:]
                recent_lengths = self.episode_lengths[-100:]
                
                info['mean_episode_reward'] = np.mean(recent_rewards)
                info['mean_episode_length'] = np.mean(recent_lengths)
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset episode tracking."""
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        return self.env.reset(**kwargs)


class TruncationWrapper(gym.Wrapper):
    """
    Add truncation logic for episodes that run too long.
    """
    
    def __init__(self, env: gym.Env, max_steps: int = 50):
        """
        Initialize truncation wrapper.
        
        Args:
            env: Environment to wrap
            max_steps: Maximum steps before truncation
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with truncation check."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_steps += 1
        
        # Check if we've exceeded max steps
        if self.current_steps >= self.max_steps and not terminated:
            truncated = True
            info['truncation_reason'] = 'max_steps_exceeded'
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset step counter."""
        self.current_steps = 0
        return self.env.reset(**kwargs)
