"""
Reward sensitivity analysis - test different reward scales.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
from src.utils.episode_builder import EpisodeBuilder
from src.utils.toxicity_judge import ToxicityJudge
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json


class RewardConfig:
    """Reward configuration for testing."""
    
    def __init__(self, name: str, 
                 engagement_reward: float = 1.0,
                 safety_penalty: float = 10.0,
                 warn_cost: float = 0.1,
                 delete_cost: float = 0.5,
                 ban_cost: float = 2.0,
                 false_positive_ban_penalty: float = 50.0,
                 true_positive_ban_bonus: float = 10.0):
        self.name = name
        self.engagement_reward = engagement_reward
        self.safety_penalty = safety_penalty
        self.warn_cost = warn_cost
        self.delete_cost = delete_cost
        self.ban_cost = ban_cost
        self.false_positive_ban_penalty = false_positive_ban_penalty
        self.true_positive_ban_bonus = true_positive_ban_bonus
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'engagement_reward': self.engagement_reward,
            'safety_penalty': self.safety_penalty,
            'warn_cost': self.warn_cost,
            'delete_cost': self.delete_cost,
            'ban_cost': self.ban_cost,
            'false_positive_ban_penalty': self.false_positive_ban_penalty,
            'true_positive_ban_bonus': self.true_positive_ban_bonus
        }


class RewardTunableEnv(DiscordEnv):
    """Environment with tunable reward parameters."""
    
    def __init__(self, reward_config: RewardConfig, **kwargs):
        super().__init__(**kwargs)
        self.reward_config = reward_config
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with custom reward configuration."""
        if self.current_episode is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Check action validity
        if not (0 <= action < self.action_space.n):
            reward = -100.0
            terminated = False
            truncated = False
            observation = self._get_observation()
            info = self._get_info()
            info['invalid_action'] = True
            return observation, reward, terminated, truncated, info
        
        valid_actions = self.action_masks()
        if not valid_actions[action]:
            reward = -100.0
            terminated = False
            truncated = False
            observation = self._get_observation()
            info = self._get_info()
            info['invalid_action'] = True
            return observation, reward, terminated, truncated, info
        
        # Get current message
        current_idx = self.current_step
        if current_idx >= len(self.current_episode['messages']):
            terminated = True
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, terminated, False, info
        
        current_message = self.current_episode['messages'][current_idx]
        current_user = current_message['user_id']
        is_toxic = current_message['toxic']
        toxicity_score = current_message['toxicity_score']
        
        # Initialize reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # === EXECUTE ACTION WITH CUSTOM REWARDS ===
        if action == self.ACTION_ALLOW:
            reward += self.reward_config.engagement_reward
            self.episode_stats['messages_allowed'] += 1
            
            if is_toxic:
                reward -= self.reward_config.safety_penalty
                self.episode_stats['safety_violations'] += 1
            
            self.server_message_count += 1
            self.server_toxicity_sum += toxicity_score
        
        elif action == self.ACTION_WARN:
            reward -= self.reward_config.warn_cost
            self.episode_stats['warnings_issued'] += 1
            self.total_moderation_actions += 1
            
            if is_toxic:
                reward += 1.0  # Keep this fixed
                self.episode_stats['true_positives'] += 1
            else:
                self.episode_stats['false_positives'] += 1
            
            self._apply_user_simulator(current_user, action)
        
        elif action == self.ACTION_DELETE:
            reward -= self.reward_config.delete_cost
            self.episode_stats['deletions'] += 1
            self.total_moderation_actions += 1
            
            if is_toxic:
                reward += 2.0  # Keep this fixed
                self.episode_stats['true_positives'] += 1
            else:
                reward -= 2.0
                self.episode_stats['false_positives'] += 1
        
        elif action == self.ACTION_BAN:
            reward -= self.reward_config.ban_cost
            self.episode_stats['bans'] += 1
            self.total_moderation_actions += 1
            
            user_features = self.current_episode['user_features'][current_idx]
            user_avg_toxicity = user_features['avg_toxicity']
            
            if user_avg_toxicity < 0.2:
                reward -= self.reward_config.false_positive_ban_penalty
                self.episode_stats['false_positives'] += 1
            elif user_avg_toxicity > 0.5:
                reward += self.reward_config.true_positive_ban_bonus
                self.episode_stats['true_positives'] += 1
            
            self.banned_users.add(current_user)
        
        # Update stats
        self.episode_stats['total_reward'] += reward
        
        self.moderation_history.append({
            'step': current_idx,
            'action': action,
            'user_id': current_user,
            'toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'reward': reward,
            'action_name': self.ACTION_NAMES[action]
        })
        
        # State transition
        self.current_step += 1
        
        while (self.current_step < len(self.current_episode['messages']) and 
               self.current_episode['messages'][self.current_step]['user_id'] in self.banned_users):
            self.current_step += 1
        
        if self.current_step >= len(self.current_episode['messages']):
            terminated = True
        
        observation = self._get_observation()
        info = self._get_info()
        info['action_name'] = self.ACTION_NAMES[action]
        info['step_reward'] = reward
        
        if terminated:
            info['episode'] = self.episode_stats.copy()
            info['final_step'] = True
        
        return observation, reward, terminated, truncated, info


def evaluate_reward_config(config: RewardConfig, n_episodes: int = 50) -> Dict:
    """
    Evaluate a reward configuration with a simple heuristic agent.
    
    Returns metrics: reward, safety violations, false positives, etc.
    """
    # Use shared components
    episode_builder = EpisodeBuilder()
    toxicity_judge = ToxicityJudge(device='cuda' if __name__ == '__main__' else 'cpu')
    
    env = RewardTunableEnv(
        reward_config=config,
        episode_builder=episode_builder,
        toxicity_judge=toxicity_judge
    )
    
    episode_rewards = []
    safety_violations = []
    false_positives = []
    true_positives = []
    messages_allowed = []
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        
        episode_reward = 0
        steps = 0
        
        while steps < 50:
            # Simple heuristic agent
            toxicity = info['toxicity_score']
            user_avg = info['user_avg_toxicity']
            
            if user_avg > 0.6:
                action = env.ACTION_BAN
            elif toxicity > 0.8:
                action = env.ACTION_DELETE
            elif toxicity > 0.5:
                action = env.ACTION_WARN
            else:
                action = env.ACTION_ALLOW
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if 'episode' in info:
            stats = info['episode']
            safety_violations.append(stats['safety_violations'])
            false_positives.append(stats['false_positives'])
            true_positives.append(stats['true_positives'])
            messages_allowed.append(stats['messages_allowed'])
    
    results = {
        'config': config.to_dict(),
        'reward_mean': float(np.mean(episode_rewards)),
        'reward_std': float(np.std(episode_rewards)),
        'safety_violations_mean': float(np.mean(safety_violations)),
        'false_positives_mean': float(np.mean(false_positives)),
        'true_positives_mean': float(np.mean(true_positives)),
        'messages_allowed_mean': float(np.mean(messages_allowed)),
        'safety_violation_rate': float(np.mean(safety_violations)) / float(np.mean(messages_allowed)) if np.mean(messages_allowed) > 0 else 0.0
    }
    
    return results


def run_sensitivity_analysis():
    """Run reward sensitivity analysis across different configurations."""
    
    print("="*60)
    print("Reward Sensitivity Analysis")
    print("="*60)
    
    # Test configurations
    configs = [
        # Baseline
        RewardConfig("Baseline", engagement_reward=1.0, safety_penalty=10.0),
        
        # Vary safety penalty
        RewardConfig("Low Safety Penalty", engagement_reward=1.0, safety_penalty=5.0),
        RewardConfig("High Safety Penalty", engagement_reward=1.0, safety_penalty=20.0),
        RewardConfig("Very High Safety", engagement_reward=1.0, safety_penalty=50.0),
        
        # Vary engagement reward
        RewardConfig("Low Engagement", engagement_reward=0.5, safety_penalty=10.0),
        RewardConfig("High Engagement", engagement_reward=2.0, safety_penalty=10.0),
        
        # Balanced configurations
        RewardConfig("Balanced 1", engagement_reward=1.5, safety_penalty=15.0),
        RewardConfig("Balanced 2", engagement_reward=2.0, safety_penalty=20.0),
        
        # Safety-first
        RewardConfig("Safety First", engagement_reward=0.5, safety_penalty=50.0),
        
        # Engagement-first
        RewardConfig("Engagement First", engagement_reward=3.0, safety_penalty=5.0),
    ]
    
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config.name}")
        print(f"  Engagement: {config.engagement_reward}, Safety: {config.safety_penalty}")
        
        results = evaluate_reward_config(config, n_episodes=30)
        all_results.append(results)
        
        print(f"  Results:")
        print(f"    Reward: {results['reward_mean']:.1f} ± {results['reward_std']:.1f}")
        print(f"    Safety violations: {results['safety_violations_mean']:.2f}")
        print(f"    False positives: {results['false_positives_mean']:.2f}")
        print(f"    Messages allowed: {results['messages_allowed_mean']:.1f}")
    
    # Save results
    output_path = Path('outputs/reward_sensitivity.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Visualize trade-offs
    visualize_tradeoffs(all_results)
    
    return all_results


def visualize_tradeoffs(results: List[Dict]):
    """Visualize safety vs engagement trade-off."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    names = [r['config']['name'] for r in results]
    rewards = [r['reward_mean'] for r in results]
    safety_viols = [r['safety_violations_mean'] for r in results]
    false_pos = [r['false_positives_mean'] for r in results]
    messages_allowed = [r['messages_allowed_mean'] for r in results]
    
    # Plot 1: Reward comparison
    axes[0, 0].barh(names, rewards, color='steelblue')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Average Episode Reward')
    axes[0, 0].set_title('Reward by Configuration')
    axes[0, 0].tick_params(axis='y', labelsize=8)
    
    # Plot 2: Safety violations
    axes[0, 1].barh(names, safety_viols, color='coral')
    axes[0, 1].set_xlabel('Avg Safety Violations per Episode')
    axes[0, 1].set_title('Safety Performance')
    axes[0, 1].tick_params(axis='y', labelsize=8)
    
    # Plot 3: Safety vs Engagement trade-off
    axes[1, 0].scatter(safety_viols, messages_allowed, s=100, alpha=0.6)
    for i, name in enumerate(names):
        axes[1, 0].annotate(name, (safety_viols[i], messages_allowed[i]), 
                           fontsize=7, ha='right')
    axes[1, 0].set_xlabel('Safety Violations')
    axes[1, 0].set_ylabel('Messages Allowed (Engagement)')
    axes[1, 0].set_title('Safety vs Engagement Trade-off')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward vs Safety violation rate
    safety_rates = [r['safety_violation_rate'] for r in results]
    axes[1, 1].scatter(safety_rates, rewards, s=100, alpha=0.6, color='green')
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (safety_rates[i], rewards[i]), 
                           fontsize=7, ha='right')
    axes[1, 1].set_xlabel('Safety Violation Rate')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].set_title('Reward vs Safety Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/reward_tradeoffs.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/reward_tradeoffs.png")
    plt.close()


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_sensitivity_analysis()
    print("\n✅ Reward sensitivity analysis complete!")
