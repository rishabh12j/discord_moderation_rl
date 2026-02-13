"""
Discord Moderation RL Environment
Gymnasium-compatible environment for training moderation agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.episode_builder import EpisodeBuilder
from utils.toxicity_judge import ToxicityJudge


class DiscordEnv(gym.Env):
    """
    Discord Moderation Environment.
    
    Observation Space:
        Dict with:
        - message_embedding: Box(384,) - Semantic content
        - user_history: Box(5,) - [avg_toxicity, total_messages, toxic_messages, 
                                    join_days_ago, recent_warnings]
        - server_stats: Box(3,) - [server_avg_toxicity, total_moderation_actions, 
                                    recent_toxicity_trend]
    
    Action Space:
        Discrete(4):
        - 0: Allow (let message through)
        - 1: Warn (warn user, message stays)
        - 2: Delete (remove message)
        - 3: Ban (ban user, end their participation)
    
    Rewards:
        Engagement reward: +1 per message allowed
        Safety cost: +10 for toxic message allowed
        Fairness penalty: -50 for banning good users
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    # Action constants
    ACTION_ALLOW = 0
    ACTION_WARN = 1
    ACTION_DELETE = 2
    ACTION_BAN = 3
    
    ACTION_NAMES = {
        0: "ALLOW",
        1: "WARN",
        2: "DELETE",
        3: "BAN"
    }
    
    def __init__(self,
                 episode_builder: Optional[EpisodeBuilder] = None,
                 toxicity_judge: Optional[ToxicityJudge] = None,
                 max_steps: int = 20,
                 render_mode: Optional[str] = None):
        """
        Initialize Discord moderation environment.
        
        Args:
            episode_builder: EpisodeBuilder instance (created if None)
            toxicity_judge: ToxicityJudge instance (created if None)
            max_steps: Maximum steps per episode (default: 20)
            render_mode: Render mode ('human' or 'ansi')
        """
        super().__init__()
        
        # Initialize components
        print("Initializing DiscordEnv...")
        
        if episode_builder is None:
            print("  Creating EpisodeBuilder...")
            self.episode_builder = EpisodeBuilder()
        else:
            self.episode_builder = episode_builder
        
        if toxicity_judge is None:
            print("  Creating ToxicityJudge...")
            self.toxicity_judge = ToxicityJudge(device="cuda", batch_size=32)
        else:
            self.toxicity_judge = toxicity_judge
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'message_embedding': spaces.Box(
                low=-1.0, high=1.0, shape=(384,), dtype=np.float32
            ),
            'user_history': spaces.Box(
                low=0.0, high=np.inf, shape=(5,), dtype=np.float32
            ),
            'server_stats': spaces.Box(
                low=0.0, high=np.inf, shape=(3,), dtype=np.float32
            )
        })
        
        # Define action space
        self.action_space = spaces.Discrete(4)
        
        # Episode state
        self.current_episode = None
        self.current_step = 0
        self.banned_users = set()
        self.user_warnings = {}  # user_id -> warning_count
        self.moderation_history = []
        
        # Server-level statistics
        self.server_toxicity_sum = 0.0
        self.server_message_count = 0
        self.total_moderation_actions = 0
        
        print("✓ DiscordEnv initialized")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to start of new episode.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Sample new episode
        self.current_episode = self.episode_builder.sample_episode()
        self.current_step = 0
        
        # Reset episode-specific state
        self.banned_users = set()
        self.user_warnings = {}
        self.moderation_history = []
        
        # Reset server stats for this episode
        self.server_toxicity_sum = 0.0
        self.server_message_count = 0
        self.total_moderation_actions = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in environment.
        
        Args:
            action: Action to take (0=Allow, 1=Warn, 2=Delete, 3=Ban)
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        if self.current_episode is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Get current message
        current_message = self.current_episode['messages'][self.current_step]
        current_user = current_message['user_id']
        is_toxic = current_message['toxic']
        toxicity_score = current_message['toxicity_score']
        
        # Initialize reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # Execute action
        if action == self.ACTION_ALLOW:
            # Allow message through
            reward += 1.0  # Engagement reward
            
            # Safety cost if toxic
            if is_toxic:
                reward -= 10.0  # Safety violation
            
            self.server_message_count += 1
            self.server_toxicity_sum += toxicity_score
        
        elif action == self.ACTION_WARN:
            # Warn user
            self.user_warnings[current_user] = self.user_warnings.get(current_user, 0) + 1
            self.total_moderation_actions += 1
            
            # Small penalty for warning (reduces engagement)
            reward -= 0.1
            
            # Apply to next message from this user (handled in _apply_user_simulator)
        
        elif action == self.ACTION_DELETE:
            # Delete message
            self.total_moderation_actions += 1
            
            # Small penalty for deletion
            reward -= 0.5
            
            # Bonus if message was actually toxic
            if is_toxic:
                reward += 2.0
        
        elif action == self.ACTION_BAN:
            # Ban user
            self.banned_users.add(current_user)
            self.total_moderation_actions += 1
            
            # Large penalty for banning
            reward -= 2.0
            
            # Check fairness: is this user actually bad?
            user_features = self.current_episode['user_features'][self.current_step]
            user_avg_toxicity = user_features['avg_toxicity']
            
            if user_avg_toxicity < 0.2:
                # Banning a good user - huge penalty
                reward -= 50.0
            elif user_avg_toxicity > 0.5:
                # Banning a troll - bonus
                reward += 10.0
        
        # Record action
        self.moderation_history.append({
            'step': self.current_step,
            'action': action,
            'user_id': current_user,
            'toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'reward': reward
        })
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.current_episode['messages']):
            terminated = True
            observation = self._get_observation()  # Final observation
        else:
            # Get next observation
            observation = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.
        
        Returns:
            Dict with message_embedding, user_history, server_stats
        """
        if self.current_step >= len(self.current_episode['messages']):
            # Episode ended - return zero observation
            return {
                'message_embedding': np.zeros(384, dtype=np.float32),
                'user_history': np.zeros(5, dtype=np.float32),
                'server_stats': np.zeros(3, dtype=np.float32)
            }
        
        # Get current message embedding
        message_embedding = self.current_episode['embeddings'][self.current_step].astype(np.float32)
        
        # Get user history
        current_message = self.current_episode['messages'][self.current_step]
        current_user = current_message['user_id']
        user_features = self.current_episode['user_features'][self.current_step]
        
        user_history = np.array([
            user_features['avg_toxicity'],
            user_features['total_messages'],
            user_features['toxic_messages'],
            user_features['join_days_ago'],
            self.user_warnings.get(current_user, 0)  # Recent warnings
        ], dtype=np.float32)
        
        # Get server stats
        avg_server_toxicity = (
            self.server_toxicity_sum / self.server_message_count 
            if self.server_message_count > 0 else 0.0
        )
        
        # Recent toxicity trend (last 5 messages)
        recent_start = max(0, self.current_step - 5)
        recent_messages = self.current_episode['messages'][recent_start:self.current_step]
        recent_toxicity = (
            np.mean([m['toxicity_score'] for m in recent_messages]) 
            if recent_messages else 0.0
        )
        
        server_stats = np.array([
            avg_server_toxicity,
            self.total_moderation_actions,
            recent_toxicity
        ], dtype=np.float32)
        
        return {
            'message_embedding': message_embedding,
            'user_history': user_history,
            'server_stats': server_stats
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            Dict with debug/analysis information
        """
        if self.current_episode is None or self.current_step >= len(self.current_episode['messages']):
            return {}
        
        current_message = self.current_episode['messages'][self.current_step]
        
        return {
            'conversation_id': self.current_episode['metadata']['conversation_id'],
            'step': self.current_step,
            'user_id': current_message['user_id'],
            'message_text': current_message['text'],
            'is_toxic': bool(current_message['toxic']),
            'toxicity_score': float(current_message['toxicity_score']),
            'total_moderation_actions': self.total_moderation_actions,
            'banned_users_count': len(self.banned_users)
        }
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            if self.current_episode is None:
                print("Environment not initialized. Call reset() first.")
                return
            
            if self.current_step > 0:
                last_action = self.moderation_history[-1]
                print(f"\n{'='*70}")
                print(f"Step {self.current_step-1}: {self.ACTION_NAMES[last_action['action']]}")
                print(f"Message: {last_action['user_id']}: "
                      f"{self.current_episode['messages'][last_action['step']]['text'][:60]}...")
                print(f"Toxic: {last_action['toxic']} | Score: {last_action['toxicity_score']:.3f}")
                print(f"Reward: {last_action['reward']:.2f}")
    
    def close(self):
        """Clean up resources."""
        pass


def main():
    """Test environment initialization."""
    print("=" * 70)
    print("Day 8: Gymnasium Environment Skeleton Test")
    print("=" * 70)
    
    # Create environment
    env = DiscordEnv()
    
    # Test observation and action spaces
    print("\n" + "=" * 70)
    print("Space Definitions")
    print("=" * 70)
    
    print("\nObservation Space:")
    print(f"  Type: {type(env.observation_space)}")
    print(f"  Keys: {list(env.observation_space.spaces.keys())}")
    for key, space in env.observation_space.spaces.items():
        print(f"    {key}: {space}")
    
    print("\nAction Space:")
    print(f"  Type: {type(env.action_space)}")
    print(f"  N: {env.action_space.n}")
    print(f"  Actions: {env.ACTION_NAMES}")
    
    # Test reset
    print("\n" + "=" * 70)
    print("Testing reset()")
    print("=" * 70)
    
    observation, info = env.reset()
    
    print("\nInitial Observation:")
    print(f"  message_embedding shape: {observation['message_embedding'].shape}")
    print(f"  user_history shape: {observation['user_history'].shape}")
    print(f"  server_stats shape: {observation['server_stats'].shape}")
    
    print("\nInitial Info:")
    for key, value in info.items():
        if key == 'message_text':
            print(f"  {key}: {value[:60]}...")
        else:
            print(f"  {key}: {value}")
    
    # Test a few steps
    print("\n" + "=" * 70)
    print("Testing step() - Random Actions")
    print("=" * 70)
    
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i}:")
        print(f"  Action: {env.ACTION_NAMES[action]}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}")
        print(f"  Message: {info['message_text'][:50]}...")
        print(f"  Toxic: {info['is_toxic']} (score: {info['toxicity_score']:.3f})")
        
        if terminated:
            print("\n  Episode ended!")
            break
    
    print("\n" + "=" * 70)
    print("✅ Day 8 Complete!")
    print("=" * 70)
    print("\nEnvironment skeleton is working!")
    print("Next: Day 9 - Implement reset() details and episode sampling")


if __name__ == "__main__":
    main()
