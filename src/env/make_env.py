"""
Environment factory for creating Discord moderation environments.
"""

import gymnasium as gym
from typing import Optional
from src.env.discord_env import DiscordEnv
from src.env.wrappers import (
    RewardNormalizationWrapper,
    EpisodeStatsWrapper,
    TruncationWrapper
)
from src.utils.episode_builder import EpisodeBuilder
from src.utils.toxicity_judge import ToxicityJudge


def make_discord_env(
    normalize_rewards: bool = True,
    track_stats: bool = True,
    truncate: bool = True,
    max_steps: int = 50,
    episode_builder: Optional[EpisodeBuilder] = None,
    toxicity_judge: Optional[ToxicityJudge] = None,
    device: str = "cuda"
) -> gym.Env:
    """
    Create a Discord moderation environment with optional wrappers.
    
    Args:
        normalize_rewards: Apply reward normalization
        track_stats: Track episode statistics
        truncate: Add truncation wrapper
        max_steps: Maximum steps per episode
        episode_builder: Pre-initialized EpisodeBuilder (creates new if None)
        toxicity_judge: Pre-initialized ToxicityJudge (creates new if None)
        device: Device for ToxicityJudge ('cpu' or 'cuda')
    
    Returns:
        Wrapped Gymnasium environment
    """
    # Create base environment
    env = DiscordEnv(
        episode_builder=episode_builder,
        toxicity_judge=toxicity_judge,
        max_steps=max_steps
    )
    
    # Apply wrappers (order matters!)
    if truncate:
        env = TruncationWrapper(env, max_steps=max_steps)
    
    if track_stats:
        env = EpisodeStatsWrapper(env)
    
    if normalize_rewards:
        env = RewardNormalizationWrapper(env)
    
    return env


def make_discord_env_vectorized(
    n_envs: int = 4,
    normalize_rewards: bool = True,
    track_stats: bool = True,
    device: str = "cuda"
) -> gym.Env:
    """
    Create vectorized Discord environments for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        normalize_rewards: Apply reward normalization
        track_stats: Track episode statistics
        device: Device for ToxicityJudge
    
    Returns:
        Vectorized environment
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    # Create shared components (loaded once, shared across envs)
    print(f"Creating {n_envs} parallel environments...")
    episode_builder = EpisodeBuilder()
    toxicity_judge = ToxicityJudge(device=device, batch_size=32)
    
    def make_env_fn():
        """Factory function for creating single environment."""
        return make_discord_env(
            normalize_rewards=normalize_rewards,
            track_stats=track_stats,
            episode_builder=episode_builder,
            toxicity_judge=toxicity_judge,
            device=device
        )
    
    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv([make_env_fn])
    else:
        # Use SubprocVecEnv for true parallelism
        env = DummyVecEnv([make_env_fn for _ in range(n_envs)])
    
    print(f"✓ Created {n_envs} parallel environments")
    return env
