"""
Evaluate baseline agent performance.

Agents to test:
1. Random Agent (baseline)
2. Always Allow (maximum engagement, zero safety)
3. Always Ban Toxic (rule-based, threshold=0.5)
4. Aggressive (low threshold=0.3)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.env.discord_env import DiscordEnv
import numpy as np
from typing import Dict, List
import json


class BaselineAgent:
    """Base class for baseline agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        """Select action given observation and info."""
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Random agent - selects random valid action."""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        valid_indices = [i for i, v in enumerate(valid_actions) if v]
        return np.random.choice(valid_indices)


class AlwaysAllowAgent(BaselineAgent):
    """Always allow messages (maximum engagement, no safety)."""
    
    def __init__(self):
        super().__init__("Always Allow")
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        return 0  # ALLOW


class RuleBasedAgent(BaselineAgent):
    """Rule-based agent: Delete if toxic > threshold."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(f"Rule-Based (threshold={threshold})")
        self.threshold = threshold
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        toxicity = info['toxicity_score']
        
        if toxicity > self.threshold:
            return 2  # DELETE
        else:
            return 0  # ALLOW


class AggressiveAgent(BaselineAgent):
    """Aggressive moderation: Warn/delete at low thresholds."""
    
    def __init__(self):
        super().__init__("Aggressive")
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        toxicity = info['toxicity_score']
        user_avg = info['user_avg_toxicity']
        
        if user_avg > 0.6:
            return 3  # BAN
        elif toxicity > 0.6:
            return 2  # DELETE
        elif toxicity > 0.3:
            return 1  # WARN
        else:
            return 0  # ALLOW


class SmartAgent(BaselineAgent):
    """Smart heuristic: Consider user profile and escalation."""
    
    def __init__(self):
        super().__init__("Smart Heuristic")
    
    def select_action(self, observation: Dict, info: Dict, valid_actions: List[int]) -> int:
        toxicity = info['toxicity_score']
        user_avg = info['user_avg_toxicity']
        user_profile = info['user_profile']
        warnings = info['warnings_this_user']
        
        # Ban trolls with repeated violations
        if user_profile == 'troll' and warnings >= 2:
            return 3  # BAN
        
        # Delete high toxicity
        if toxicity > 0.8:
            return 2  # DELETE
        
        # Warn borderline
        if toxicity > 0.5:
            return 1  # WARN
        
        # Allow good users even with slight toxicity
        if user_profile == 'good_user' and toxicity < 0.4:
            return 0  # ALLOW
        
        # Default: warn if somewhat toxic
        if toxicity > 0.4:
            return 1  # WARN
        
        return 0  # ALLOW


def evaluate_agent(agent: BaselineAgent, n_episodes: int = 50) -> Dict:
    """
    Evaluate agent over multiple episodes.
    
    Returns:
        Dict with comprehensive metrics
    """
    env = DiscordEnv()
    
    episode_rewards = []
    episode_stats = []
    
    print(f"\nEvaluating {agent.name}...")
    print(f"Running {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        
        episode_reward = 0
        steps = 0
        
        while steps < 50:
            valid_actions = info['valid_actions']
            action = agent.select_action(observation, info, valid_actions)
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if 'episode' in info:
            episode_stats.append(info['episode'])
        
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode + 1}/{n_episodes} episodes")
    
    # Aggregate statistics
    results = {
        'agent_name': agent.name,
        'n_episodes': n_episodes,
        'rewards': {
            'mean': float(np.mean(episode_rewards)),
            'std': float(np.std(episode_rewards)),
            'min': float(np.min(episode_rewards)),
            'max': float(np.max(episode_rewards))
        }
    }
    
    if episode_stats:
        # Aggregate episode statistics
        results['safety_violations'] = {
            'mean': float(np.mean([s['safety_violations'] for s in episode_stats])),
            'total': int(np.sum([s['safety_violations'] for s in episode_stats]))
        }
        results['true_positives'] = {
            'mean': float(np.mean([s['true_positives'] for s in episode_stats])),
            'total': int(np.sum([s['true_positives'] for s in episode_stats]))
        }
        results['false_positives'] = {
            'mean': float(np.mean([s['false_positives'] for s in episode_stats])),
            'total': int(np.sum([s['false_positives'] for s in episode_stats]))
        }
        results['messages_allowed'] = {
            'mean': float(np.mean([s['messages_allowed'] for s in episode_stats]))
        }
        results['warnings_issued'] = {
            'mean': float(np.mean([s['warnings_issued'] for s in episode_stats]))
        }
        results['deletions'] = {
            'mean': float(np.mean([s['deletions'] for s in episode_stats]))
        }
        results['bans'] = {
            'mean': float(np.mean([s['bans'] for s in episode_stats]))
        }
        
        # Calculate precision and recall if possible
        tp = results['true_positives']['total']
        fp = results['false_positives']['total']
        
        if tp + fp > 0:
            results['precision'] = tp / (tp + fp)
        else:
            results['precision'] = 0.0
    
    return results


def compare_baselines():
    """Compare all baseline agents."""
    agents = [
        RandomAgent(),
        AlwaysAllowAgent(),
        RuleBasedAgent(threshold=0.5),
        RuleBasedAgent(threshold=0.7),
        AggressiveAgent(),
        SmartAgent()
    ]
    
    all_results = []
    
    print("="*60)
    print("Baseline Agent Comparison")
    print("="*60)
    
    for agent in agents:
        results = evaluate_agent(agent, n_episodes=50)
        all_results.append(results)
    
    # Print comparison table
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    print(f"\n{'Agent':<25} {'Reward':<15} {'Safety Viol.':<15} {'Precision':<10}")
    print("-"*65)
    
    for results in all_results:
        agent_name = results['agent_name']
        reward = results['rewards']['mean']
        safety = results.get('safety_violations', {}).get('mean', 0)
        precision = results.get('precision', 0) * 100
        
        print(f"{agent_name:<25} {reward:>7.1f} ± {results['rewards']['std']:>4.1f}  "
              f"{safety:>6.1f}          {precision:>6.1f}%")
    
    # Save results
    output_path = Path('outputs/baseline_results.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    compare_baselines()
    print("\n✅ Baseline evaluation complete!")
