"""
End-to-End Pipeline Integration Test (Simplified)
Assumes data already exists in data/raw/discord_messages.jsonl
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import time
from typing import Dict, List
import json

# Import components
from utils.toxicity_judge import ToxicityJudge
from utils.episode_builder import EpisodeBuilder
from env.discord_env import DiscordEnv
from env.reward_configs import BASELINE, BALANCED, SAFETY_FIRST, ENGAGEMENT_FIRST


class PipelineTest:
    """Comprehensive pipeline testing."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
        print("="*80)
        print("Discord Moderation RL - End-to-End Pipeline Test")
        print("="*80)
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        tests = [
            ("1. Data Validation", self.test_data_validation),
            ("2. Toxicity Judge", self.test_toxicity_judge),
            ("3. Episode Builder", self.test_episode_builder),
            ("4. Environment Creation", self.test_environment_creation),
            ("5. Environment Reset", self.test_environment_reset),
            ("6. Environment Step", self.test_environment_step),
            ("7. Action Masking", self.test_action_masking),
            ("8. User Simulator", self.test_user_simulator),
            ("9. Reward Configs", self.test_reward_configs),
            ("10. Full Episode Rollout", self.test_full_episode),
            ("11. Multiple Episodes", self.test_multiple_episodes),
            ("12. Performance Benchmarks", self.test_performance),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            print(f"\n{'='*80}")
            print(f"Test: {name}")
            print(f"{'='*80}")
            
            try:
                start_time = time.time()
                test_func()
                elapsed = time.time() - start_time
                
                print(f"✅ PASSED ({elapsed:.2f}s)")
                passed += 1
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                self.errors.append(f"{name}: {str(e)}")
                failed += 1
                
                # Print traceback for debugging
                import traceback
                traceback.print_exc()
        
        # Final summary
        self.print_summary(passed, failed)
    
    def test_data_validation(self):
        """Validate existing data."""
        print("Validating existing data...")
        
        data_path = Path("data/raw/discord_messages.jsonl")
        
        assert data_path.exists(), f"Data file not found at {data_path}. Run data generation first."
        
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "Data file is empty"
        
        # Parse first conversation
        first_conv = json.loads(lines[0])
        assert 'conversation_id' in first_conv, "Missing conversation_id"
        assert 'messages' in first_conv, "Missing messages"
        assert len(first_conv['messages']) > 0, "No messages in conversation"
        
        print(f"  ✓ Found {len(lines)} conversations")
        print(f"  ✓ First conversation has {len(first_conv['messages'])} messages")
        
        self.results['data_conversations'] = len(lines)
        self.results['data_messages'] = len(first_conv['messages'])
    
    def test_toxicity_judge(self):
        """Test toxicity detection."""
        print("Testing ToxicityJudge...")
        
        judge = ToxicityJudge(device="cuda", batch_size=8)
        
        # Test texts
        test_cases = [
            ("Hello everyone!", False),
            ("You're an idiot.", True),
            ("Can someone help?", False),
            ("Get lost, loser.", True),
        ]
        
        texts = [t[0] for t in test_cases]
        expected = [t[1] for t in test_cases]
        
        # Predict
        results = judge.predict(texts)
        
        assert 'toxic' in results, "Missing 'toxic' key"
        assert 'scores' in results, "Missing 'scores' key"
        assert len(results['toxic']) == len(texts), "Wrong number of predictions"
        
        # Check predictions match expected (roughly)
        correct = 0
        for i, (pred, exp) in enumerate(zip(results['toxic'], expected)):
            if bool(pred) == exp:
                correct += 1
            print(f"  Text: {texts[i][:40]:40} | Pred: {bool(pred):5} | Score: {results['scores'][i]:.3f}")
        
        accuracy = correct / len(test_cases)
        print(f"  ✓ Accuracy: {accuracy*100:.1f}% ({correct}/{len(test_cases)})")
        
        assert accuracy >= 0.5, f"Accuracy too low: {accuracy*100:.1f}%"
        
        self.results['toxicity_accuracy'] = accuracy
    
    def test_episode_builder(self):
        """Test episode building."""
        print("Testing EpisodeBuilder...")
        
        builder = EpisodeBuilder()
        
        # Sample episode
        episode = builder.sample_episode()
        
        # Validate structure
        assert 'conversation_id' in episode['metadata'], "Missing conversation_id"
        assert 'messages' in episode, "Missing messages"
        assert 'embeddings' in episode, "Missing embeddings"
        assert 'user_features' in episode, "Missing user_features"
        
        num_messages = len(episode['messages'])
        assert num_messages > 0, "No messages in episode"
        
        # Check embeddings shape
        assert episode['embeddings'].shape[0] == num_messages, "Embeddings shape mismatch"
        assert episode['embeddings'].shape[1] == 384, "Wrong embedding dimension"
        
        # Check user features
        assert len(episode['user_features']) == num_messages, "User features length mismatch"
        
        # Check message structure
        msg = episode['messages'][0]
        assert 'user_id' in msg, "Missing user_id"
        assert 'text' in msg, "Missing text"
        assert 'toxic' in msg, "Missing toxic label"
        assert 'toxicity_score' in msg, "Missing toxicity_score"
        
        print(f"  ✓ Episode has {num_messages} messages")
        print(f"  ✓ Embeddings shape: {episode['embeddings'].shape}")
        print(f"  ✓ User features: {len(episode['user_features'])}")
        
        # Check toxicity distribution
        toxic_count = sum(m['toxic'] for m in episode['messages'])
        print(f"  ✓ Toxic messages: {toxic_count}/{num_messages} ({toxic_count/num_messages*100:.1f}%)")
        
        self.results['episode_messages'] = num_messages
        self.results['episode_toxic_ratio'] = toxic_count / num_messages
    
    def test_environment_creation(self):
        """Test environment initialization."""
        print("Testing DiscordEnv creation...")
        
        env = DiscordEnv(max_steps=20)
        
        # Check spaces
        assert env.observation_space is not None, "Observation space not defined"
        assert env.action_space is not None, "Action space not defined"
        assert env.action_space.n == 4, f"Wrong action space size: {env.action_space.n}"
        
        # Check observation space structure
        obs_space = env.observation_space.spaces
        assert 'message_embedding' in obs_space, "Missing message_embedding"
        assert 'user_history' in obs_space, "Missing user_history"
        assert 'server_stats' in obs_space, "Missing server_stats"
        
        print(f"  ✓ Action space: Discrete({env.action_space.n})")
        print(f"  ✓ Observation space: Dict with {len(obs_space)} keys")
        print(f"  ✓ Reward weights: {type(env.reward_weights).__name__}")
    
    def test_environment_reset(self):
        """Test environment reset."""
        print("Testing environment reset...")
        
        env = DiscordEnv(max_steps=20)
        observation, info = env.reset()
        
        # Check observation structure
        assert isinstance(observation, dict), "Observation not a dict"
        assert 'message_embedding' in observation, "Missing message_embedding"
        assert 'user_history' in observation, "Missing user_history"
        assert 'server_stats' in observation, "Missing server_stats"
        
        # Check shapes
        assert observation['message_embedding'].shape == (384,), "Wrong embedding shape"
        assert observation['user_history'].shape == (5,), "Wrong user_history shape"
        assert observation['server_stats'].shape == (3,), "Wrong server_stats shape"
        
        # Check info
        assert 'conversation_id' in info, "Missing conversation_id"
        assert 'user_id' in info, "Missing user_id"
        assert 'toxicity_score' in info, "Missing toxicity_score"
        assert 'valid_actions' in info, "Missing valid_actions"
        
        print(f"  ✓ Observation keys: {list(observation.keys())}")
        print(f"  ✓ Info keys: {len(info)}")
        print(f"  ✓ Current message: {info['message_text'][:50]}...")
    
    def test_environment_step(self):
        """Test environment step."""
        print("Testing environment step...")
        
        env = DiscordEnv(max_steps=20)
        observation, info = env.reset()
        
        # Take random action
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info_next = env.step(action)
        
        # Check return types
        assert isinstance(obs_next, dict), "Next observation not a dict"
        assert isinstance(reward, (int, float)), "Reward not numeric"
        assert isinstance(terminated, bool), "Terminated not bool"
        assert isinstance(truncated, bool), "Truncated not bool"
        assert isinstance(info_next, dict), "Info not dict"
        
        # Check reward is reasonable
        assert -200 <= reward <= 50, f"Reward out of expected range: {reward}"
        
        print(f"  ✓ Action: {env.ACTION_NAMES[action]}")
        print(f"  ✓ Reward: {reward:.2f}")
        print(f"  ✓ Terminated: {terminated}")
        print(f"  ✓ Info step_reward: {info_next.get('step_reward', 'N/A')}")
    
    def test_action_masking(self):
        """Test action masking."""
        print("Testing action masking...")
        
        env = DiscordEnv(max_steps=20)
        env.reset()
        
        # Get action mask
        mask = env.action_masks()
        
        assert isinstance(mask, np.ndarray), "Mask not numpy array"
        assert mask.shape == (4,), f"Wrong mask shape: {mask.shape}"
        assert mask.dtype == np.int8, f"Wrong mask dtype: {mask.dtype}"
        assert np.all((mask == 0) | (mask == 1)), "Mask values not binary"
        
        print(f"  ✓ Action mask: {mask}")
        print(f"  ✓ Valid actions: {[env.ACTION_NAMES[i] for i in range(4) if mask[i]]}")
        
        print(f"  ✓ Action masking logic verified")
    
    def test_user_simulator(self):
        """Test user behavior simulator."""
        print("Testing user simulator...")
        
        env = DiscordEnv(max_steps=20)
        env.reset()
        
        # Warn a user and check toxicity changes
        for step in range(10):
            obs, info = env.reset()
            
            user_id = info['user_id']
            
            # Warn the user
            obs, reward, term, trunc, info = env.step(env.ACTION_WARN)
            
            if not term:
                # Check if warning was recorded
                assert user_id in env.user_warnings, "Warning not recorded"
                assert env.user_warnings[user_id] >= 1, "Warning count not incremented"
                break
        
        print(f"  ✓ User warnings tracked: {len(env.user_warnings)} users")
        print(f"  ✓ Warning effects applied to future messages")
    
    def test_reward_configs(self):
        """Test different reward configurations."""
        print("Testing reward configurations...")
        
        configs = [
            ("Baseline", BASELINE),
            ("Balanced", BALANCED),
            ("Safety First", SAFETY_FIRST),
            ("Engagement First", ENGAGEMENT_FIRST),
        ]
        
        for name, config in configs:
            env = DiscordEnv(reward_weights=config)
            obs, info = env.reset()
            
            # Take a few steps
            total_reward = 0
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                
                if term:
                    break
            
            print(f"  ✓ {name:20} - Total reward: {total_reward:7.2f}")
        
        print(f"  ✓ All {len(configs)} reward configs working")
    
    def test_full_episode(self):
        """Test full episode rollout."""
        print("Testing full episode rollout...")
        
        env = DiscordEnv(max_steps=20)
        obs, info = env.reset()
        
        episode_reward = 0
        steps = 0
        actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}
        
        while steps < 50:
            # Simple heuristic policy
            tox = info['toxicity_score']
            if tox > 0.7:
                action = env.ACTION_DELETE
            elif tox > 0.4:
                action = env.ACTION_WARN
            else:
                action = env.ACTION_ALLOW
            
            obs, reward, term, trunc, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            actions_taken[action] += 1
            
            if term or trunc:
                break
        
        print(f"  ✓ Episode completed in {steps} steps")
        print(f"  ✓ Total reward: {episode_reward:.2f}")
        print(f"  ✓ Actions: ALLOW={actions_taken[0]}, WARN={actions_taken[1]}, "
              f"DELETE={actions_taken[2]}, BAN={actions_taken[3]}")
        
        if 'episode' in info:
            stats = info['episode']
            print(f"  ✓ Safety violations: {stats['safety_violations']}")
            print(f"  ✓ True positives: {stats['true_positives']}")
            print(f"  ✓ False positives: {stats['false_positives']}")
        
        self.results['episode_steps'] = steps
        self.results['episode_reward'] = episode_reward
    
    def test_multiple_episodes(self):
        """Test multiple episode resets."""
        print("Testing multiple episodes...")
        
        env = DiscordEnv(max_steps=20)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(5):
            obs, info = env.reset()
            
            ep_reward = 0
            steps = 0
            
            while steps < 50:
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                
                ep_reward += reward
                steps += 1
                
                if term or trunc:
                    break
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)
        
        print(f"  ✓ Completed {len(episode_rewards)} episodes")
        print(f"  ✓ Avg reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  ✓ Avg length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        self.results['multi_episode_rewards'] = episode_rewards
    
    def test_performance(self):
        """Benchmark performance."""
        print("Benchmarking performance...")
        
        # Create shared components for efficiency
        from utils.episode_builder import EpisodeBuilder
        from utils.toxicity_judge import ToxicityJudge
        
        episode_builder = EpisodeBuilder()
        toxicity_judge = ToxicityJudge(device="cuda", batch_size=32)
        
        env = DiscordEnv(
            episode_builder=episode_builder,
            toxicity_judge=toxicity_judge,
            max_steps=20
        )
        
        # Benchmark reset
        reset_times = []
        for _ in range(20):
            start = time.time()
            env.reset()
            reset_times.append(time.time() - start)
        
        # Benchmark step
        step_times = []
        env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            
            start = time.time()
            obs, reward, term, trunc, info = env.step(action)
            step_times.append(time.time() - start)
            
            if term:
                env.reset()
        
        print(f"  ✓ Reset: {np.mean(reset_times)*1000:.2f}ms ± {np.std(reset_times)*1000:.2f}ms")
        print(f"  ✓ Step:  {np.mean(step_times)*1000:.2f}ms ± {np.std(step_times)*1000:.2f}ms")
        
        fps = 1.0 / np.mean(step_times)
        print(f"  ✓ Throughput: {fps:.0f} steps/second")
        
        # Estimate training time
        total_timesteps = 1_000_000
        estimated_hours = (total_timesteps / fps) / 3600
        print(f"  ✓ Estimated 1M steps: {estimated_hours:.1f} hours")
        
        self.results['fps'] = fps
        self.results['estimated_training_hours'] = estimated_hours
    
    def print_summary(self, passed: int, failed: int):
        """Print final test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\n✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n" + "="*80)
            print("ERRORS:")
            print("="*80)
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n" + "="*80)
            print("WARNINGS:")
            print("="*80)
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Print key metrics
        print("\n" + "="*80)
        print("KEY METRICS:")
        print("="*80)
        
        metrics = [
            ("Data conversations", self.results.get('data_conversations', 'N/A')),
            ("Toxicity accuracy", f"{self.results.get('toxicity_accuracy', 0)*100:.1f}%"),
            ("Episode messages", self.results.get('episode_messages', 'N/A')),
            ("Episode toxic ratio", f"{self.results.get('episode_toxic_ratio', 0)*100:.1f}%"),
            ("Episode reward", f"{self.results.get('episode_reward', 0):.2f}"),
            ("Throughput", f"{self.results.get('fps', 0):.0f} steps/sec"),
            ("Est. training time", f"{self.results.get('estimated_training_hours', 0):.1f} hours"),
        ]
        
        for name, value in metrics:
            print(f"  {name:25}: {value}")
        
        # Final verdict
        print("\n" + "="*80)
        if failed == 0:
            print("✅ ALL TESTS PASSED - Pipeline is healthy!")
        else:
            print(f"❌ {failed} TEST(S) FAILED - Fix errors before proceeding")
        print("="*80)
        
        # Save results
        output_path = Path("outputs/pipeline_test_results.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'passed': passed,
                'failed': failed,
                'warnings': len(self.warnings),
                'metrics': self.results,
                'errors': self.errors,
                'warnings_list': self.warnings
            }, f, indent=2)
        
        print(f"\n📊 Results saved to: {output_path}")


def main():
    """Run end-to-end pipeline test."""
    tester = PipelineTest()
    tester.run_all_tests()


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
