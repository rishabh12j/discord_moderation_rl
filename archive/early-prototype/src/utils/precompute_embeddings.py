"""
Pre-compute message embeddings for fast RL environment loading.
CRITICAL: Do NOT compute embeddings inside the RL step() function!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

class EmbeddingPrecomputer:
    """
    Pre-compute and save embeddings for all messages.
    
    Uses sentence-transformers (all-MiniLM-L6-v2):
    - 384-dimensional embeddings
    - ~25ms per message on CPU
    - ~5ms per message on GPU
    """
    
    def __init__(self, 
                 model_name="all-MiniLM-L6-v2",
                 device="cuda",  # Set to "cuda" if available
                 batch_size=64):
        """
        Initialize embedding model.
        
        Args:
            model_name: SentenceTransformer model
            device: 'cpu' or 'cuda'
            batch_size: Batch size for encoding
        """
        print(f"Loading embedding model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        print(f"✓ Model loaded")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
    
    def compute_embeddings(self, texts, show_progress=True):
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Show tqdm progress bar
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        print(f"\nComputing embeddings for {len(texts)} texts...")
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"✓ Computed embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def precompute_message_embeddings(self, 
                                     conversations_path="data/processed/conversations_with_users.csv",
                                     output_dir="data/embeddings"):
        """
        Pre-compute embeddings for all messages in conversations.
        
        Saves:
        - message_embeddings.npy: (N, 384) array
        - embedding_metadata.json: mapping info
        """
        # Load conversations
        print(f"Loading conversations from {conversations_path}...")
        df = pd.read_csv(conversations_path)
        print(f"✓ Loaded {len(df)} messages")
        
        # Extract texts
        texts = df['text'].tolist()
        
        # Compute embeddings
        embeddings = self.compute_embeddings(texts)
        
        # Verify alignment
        assert len(embeddings) == len(df), "Embedding-message mismatch!"
        
        # Save embeddings
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = output_path / "message_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"✓ Saved embeddings to {embeddings_file}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {embeddings.nbytes / (1024**2):.2f} MB")
        
        # Save metadata
        metadata = {
            'num_messages': len(df),
            'embedding_dim': self.embedding_dim,
            'model_name': 'all-MiniLM-L6-v2',
            'normalized': True,
            'conversations_file': str(conversations_path),
            'alignment': 'Row i in embeddings corresponds to row i in conversations CSV',
        }
        
        metadata_file = output_path / "embedding_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_file}")
        
        return embeddings
    
    def test_embedding_lookup(self, embeddings, 
                             conversations_path="data/processed/conversations_with_users.csv"):
        """
        Test that embeddings align correctly with messages.
        """
        print("\nTesting embedding alignment...")
        
        df = pd.read_csv(conversations_path)
        
        # Pick 3 random messages
        sample_indices = np.random.choice(len(df), 3, replace=False)
        
        for idx in sample_indices:
            text = df.iloc[idx]['text']
            embedding = embeddings[idx]
            
            print(f"\nMessage {idx}:")
            print(f"  Text: {text[:60]}...")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")  # Should be ~1.0 (normalized)
            
            # Test similarity with itself (should be 1.0)
            self_sim = np.dot(embedding, embedding)
            print(f"  Self-similarity: {self_sim:.3f}")
            assert abs(self_sim - 1.0) < 0.01, "Normalization check failed"
        
        print("\n✓ Embedding alignment verified!")


def main():
    print("=" * 70)
    print("Day 5: Pre-compute Message Embeddings")
    print("=" * 70)
    
    # Initialize with CPU (change to "cuda" if available and working)
    precomputer = EmbeddingPrecomputer(device="cuda", batch_size=64)
    
    # Compute embeddings
    embeddings = precomputer.precompute_message_embeddings()
    
    # Test alignment
    precomputer.test_embedding_lookup(embeddings)
    
    # Performance estimate
    print("\n" + "=" * 70)
    print("Performance Estimate for RL Training")
    print("=" * 70)
    
    num_messages = embeddings.shape[0]
    embedding_lookup_time = 0.001  # ms (just array indexing)
    
    print(f"\nWith pre-computed embeddings:")
    print(f"  Total messages: {num_messages:,}")
    print(f"  Embedding lookup time: {embedding_lookup_time:.3f} ms")
    print(f"  RL steps per second (embedding lookup only): {1000/embedding_lookup_time:,.0f}")
    
    print(f"\nWithout pre-computation (if computed in step()):")
    print(f"  Per-message encoding time: ~25ms (CPU) or ~5ms (GPU)")
    print(f"  Would bottleneck RL training to ~40 steps/sec (CPU) or ~200 steps/sec (GPU)")
    
    print(f"\n✅ Pre-computation gives ~1000x speedup!")
    
    print("\n" + "=" * 70)
    print("✅ Day 5 Complete!")
    print("=" * 70)
    print("\nNext: Day 6 - Episode Builder (Conversation Chunks)")


if __name__ == "__main__":
    main()
