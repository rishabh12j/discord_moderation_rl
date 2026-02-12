"""
Test episode builder.
"""
import sys
sys.path.append('src')

from utils.episode_builder import EpisodeBuilder

import numpy as np

def test_initialization():
    """Test episode builder initializes correctly."""
    builder = EpisodeBuilder()
    
    assert len(builder.conversation_ids) > 0
    assert builder.embeddings.shape[1] == 384
    assert len(builder.user_lookup) > 0
    
    print("✓ Initialization test passed")


def test_episode_structure():
    """Test episode has correct structure."""
    builder = EpisodeBuilder()
    episode = builder.sample_episode()
    
    # Check keys
    assert 'messages' in episode
    assert 'embeddings' in episode
    assert 'user_features' in episode
    assert 'metadata' in episode
    
    # Check lengths match
    assert len(episode['messages']) == builder.messages_per_episode
    assert len(episode['embeddings']) == builder.messages_per_episode
    assert len(episode['user_features']) == builder.messages_per_episode
    
    # Check embedding shape
    assert episode['embeddings'].shape == (builder.messages_per_episode, 384)
    
    print("✓ Episode structure test passed")


def test_batch_sampling():
    """Test batch sampling."""
    builder = EpisodeBuilder()
    batch = builder.sample_batch(batch_size=10)
    
    assert len(batch) == 10
    
    for episode in batch:
        assert len(episode['messages']) == builder.messages_per_episode
    
    print("✓ Batch sampling test passed")


def test_train_test_split():
    """Test train/test split."""
    builder = EpisodeBuilder()
    train_ids, test_ids = builder.get_train_test_split(test_ratio=0.2)
    
    # Check no overlap
    assert len(set(train_ids) & set(test_ids)) == 0
    
    # Check ratio
    total = len(train_ids) + len(test_ids)
    assert abs(len(test_ids) / total - 0.2) < 0.01
    
    print("✓ Train/test split test passed")


if __name__ == "__main__":
    print("Running episode builder tests...\n")
    test_initialization()
    test_episode_structure()
    test_batch_sampling()
    test_train_test_split()
    print("\n✅ All episode builder tests passed!")
