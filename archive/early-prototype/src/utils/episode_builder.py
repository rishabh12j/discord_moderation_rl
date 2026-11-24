"""
Episode Builder - Generates conversation chunks for RL episodes.
Pre-loads embeddings and user features for fast access during training.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

class EpisodeBuilder:
    """
    Builds RL episodes from conversation data.
    
    An episode = one conversation thread (20 messages).
    Pre-loads all data for O(1) lookups during training.
    """
    
    def __init__(self,
                 conversations_path="data/processed/conversations_with_users.csv",
                 embeddings_path="data/embeddings/message_embeddings.npy",
                 user_lookup_path="data/processed/user_lookup.json",
                 messages_per_episode=20):
        """
        Initialize episode builder with pre-loaded data.
        
        Args:
            conversations_path: Path to conversations CSV
            embeddings_path: Path to pre-computed embeddings
            user_lookup_path: Path to user feature lookup
            messages_per_episode: Messages per episode (default 20)
        """
        print("=" * 70)
        print("Initializing Episode Builder")
        print("=" * 70)
        
        # Load conversations
        print(f"\n[1/3] Loading conversations from {conversations_path}...")
        self.conversations = pd.read_csv(conversations_path)
        print(f"✓ Loaded {len(self.conversations)} messages")
        
        # Load embeddings
        print(f"\n[2/3] Loading pre-computed embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings: shape {self.embeddings.shape}")
        
        # Verify alignment
        assert len(self.embeddings) == len(self.conversations), \
            f"Embedding-conversation mismatch: {len(self.embeddings)} != {len(self.conversations)}"
        
        # Load user lookup
        print(f"\n[3/3] Loading user features from {user_lookup_path}...")
        with open(user_lookup_path) as f:
            self.user_lookup = json.load(f)
        print(f"✓ Loaded {len(self.user_lookup)} user profiles")
        
        # Build conversation index for fast sampling
        self.messages_per_episode = messages_per_episode
        self._build_conversation_index()
        
        print("\n" + "=" * 70)
        print("✅ Episode Builder Ready!")
        print("=" * 70)
        print(f"\nTotal episodes available: {len(self.conversation_ids)}")
        print(f"Messages per episode: {self.messages_per_episode}")
        print(f"Total messages: {len(self.conversations)}")
    
    def _build_conversation_index(self):
        """Build index of valid conversation IDs."""
        print("\nBuilding conversation index...")
        
        # Group by conversation_id
        conv_groups = self.conversations.groupby('conversation_id')
        
        # Filter conversations with exactly messages_per_episode messages
        self.conversation_ids = [
            conv_id for conv_id, group in conv_groups 
            if len(group) == self.messages_per_episode
        ]
        
        print(f"✓ Indexed {len(self.conversation_ids)} valid conversations")
        
        # Calculate statistics
        toxic_counts = []
        for conv_id in self.conversation_ids:
            conv_data = self.conversations[self.conversations['conversation_id'] == conv_id]
            toxic_counts.append(conv_data['toxic'].sum())
        
        toxic_counts = np.array(toxic_counts)
        print(f"  Average toxic messages per conversation: {toxic_counts.mean():.1f}")
        print(f"  Conversations with >10 toxic messages: {(toxic_counts > 10).sum()}")
    
    def get_episode(self, conversation_id: int) -> Dict:
        """
        Get a complete episode (conversation) by ID.
        
        Args:
            conversation_id: Conversation ID to retrieve
            
        Returns:
            Dict containing:
            - messages: List of message dicts
            - embeddings: np.ndarray of shape (N, 384)
            - user_features: List of user feature dicts
            - metadata: Episode metadata
        """
        # Get conversation messages
        conv_mask = self.conversations['conversation_id'] == conversation_id
        conv_df = self.conversations[conv_mask].sort_values('message_id').reset_index(drop=True)
        
        if len(conv_df) == 0:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get embeddings for this conversation
        conv_indices = conv_df.index.tolist()
        conv_embeddings = self.embeddings[conv_indices]
        
        # Build message list
        messages = []
        user_features = []
        
        for idx, row in conv_df.iterrows():
            # Message data
            message = {
                'message_id': int(row['message_id']),
                'text': row['text'],
                'user_id': row['user_id'],
                'toxic': int(row['toxic']),
                'toxicity_score': float(row['toxicity_score']),
                'timestamp': int(row['timestamp']),
            }
            messages.append(message)
            
            # User features
            user_id = row['user_id']
            if user_id in self.user_lookup:
                user_features.append(self.user_lookup[user_id])
            else:
                # Default features for unknown users
                user_features.append({
                    'avg_toxicity': 0.05,
                    'total_messages': 1,
                    'toxic_messages': 0,
                    'profile': 'new_user',
                    'join_days_ago': 0,
                })
        
        # Metadata
        metadata = {
            'conversation_id': int(conversation_id),
            'num_messages': len(messages),
            'total_toxicity': sum(m['toxic'] for m in messages),
            'avg_toxicity_score': float(conv_df['toxicity_score'].mean()),
        }
        
        return {
            'messages': messages,
            'embeddings': conv_embeddings,
            'user_features': user_features,
            'metadata': metadata,
        }
    
    def sample_episode(self) -> Dict:
        """
        Sample a random episode.
        
        Returns:
            Episode dict (same structure as get_episode)
        """
        conversation_id = random.choice(self.conversation_ids)
        return self.get_episode(conversation_id)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Sample a batch of episodes.
        
        Args:
            batch_size: Number of episodes to sample
            
        Returns:
            List of episode dicts
        """
        conversation_ids = random.choices(self.conversation_ids, k=batch_size)
        return [self.get_episode(conv_id) for conv_id in conversation_ids]
    
    def get_train_test_split(self, test_ratio=0.2, seed=42) -> Tuple[List[int], List[int]]:
        """
        Split conversation IDs into train/test sets.
        
        Args:
            test_ratio: Fraction for test set
            seed: Random seed
            
        Returns:
            (train_ids, test_ids)
        """
        random.seed(seed)
        shuffled_ids = self.conversation_ids.copy()
        random.shuffle(shuffled_ids)
        
        split_idx = int(len(shuffled_ids) * (1 - test_ratio))
        train_ids = shuffled_ids[:split_idx]
        test_ids = shuffled_ids[split_idx:]
        
        print(f"\nTrain/Test Split:")
        print(f"  Train: {len(train_ids)} conversations")
        print(f"  Test: {len(test_ids)} conversations")
        
        return train_ids, test_ids
    
    def iterate_episodes(self, conversation_ids: Optional[List[int]] = None, shuffle=True):
        """
        Generator that yields episodes.
        
        Args:
            conversation_ids: List of conversation IDs to iterate over (default: all)
            shuffle: Shuffle order
            
        Yields:
            Episode dicts
        """
        if conversation_ids is None:
            conversation_ids = self.conversation_ids.copy()
        
        if shuffle:
            random.shuffle(conversation_ids)
        
        for conv_id in conversation_ids:
            yield self.get_episode(conv_id)


def main():
    """Test episode builder."""
    print("=" * 70)
    print("Day 6: Episode Builder Test")
    print("=" * 70)
    
    # Initialize builder
    builder = EpisodeBuilder()
    
    # Test single episode retrieval
    print("\n" + "=" * 70)
    print("Testing Single Episode Retrieval")
    print("=" * 70)
    
    episode = builder.sample_episode()
    
    print(f"\nEpisode Metadata:")
    print(f"  Conversation ID: {episode['metadata']['conversation_id']}")
    print(f"  Number of messages: {episode['metadata']['num_messages']}")
    print(f"  Total toxic: {episode['metadata']['total_toxicity']}")
    print(f"  Avg toxicity score: {episode['metadata']['avg_toxicity_score']:.3f}")
    
    print(f"\nFirst 3 messages:")
    for i, msg in enumerate(episode['messages'][:3]):
        print(f"\n  Message {i}:")
        print(f"    User: {msg['user_id']}")
        print(f"    Text: {msg['text'][:60]}...")
        print(f"    Toxic: {msg['toxic']} (score: {msg['toxicity_score']:.3f})")
        print(f"    User profile: {episode['user_features'][i]['profile']}")
    
    print(f"\nEmbeddings shape: {episode['embeddings'].shape}")
    
    # Test batch sampling
    print("\n" + "=" * 70)
    print("Testing Batch Sampling")
    print("=" * 70)
    
    batch = builder.sample_batch(batch_size=5)
    print(f"\nSampled {len(batch)} episodes")
    
    toxicity_rates = [ep['metadata']['avg_toxicity_score'] for ep in batch]
    print(f"Toxicity scores: {[f'{t:.3f}' for t in toxicity_rates]}")
    
    # Test train/test split
    print("\n" + "=" * 70)
    print("Testing Train/Test Split")
    print("=" * 70)
    
    train_ids, test_ids = builder.get_train_test_split(test_ratio=0.2)
    
    # Test iteration
    print("\n" + "=" * 70)
    print("Testing Episode Iteration")
    print("=" * 70)
    
    print("\nIterating first 3 training episodes:")
    for i, episode in enumerate(builder.iterate_episodes(train_ids[:3])):
        print(f"  Episode {i}: conv_id={episode['metadata']['conversation_id']}, "
              f"toxic={episode['metadata']['total_toxicity']}")
    
    print("\n" + "=" * 70)
    print("✅ Day 6 Complete!")
    print("=" * 70)
    print("\nEpisode Builder is ready for RL training!")
    print("Next: Day 7 - Gymnasium Environment Skeleton")


if __name__ == "__main__":
    main()
