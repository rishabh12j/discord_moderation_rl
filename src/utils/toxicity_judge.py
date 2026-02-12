"""
Toxicity Judge - A wrapper around unitary/unbiased-toxic-roberta
Provides fast toxicity scoring for moderation decisions.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Union
import time
from pathlib import Path

class ToxicityJudge:
    """
    Fast toxicity classifier for content moderation.
    
    Uses unitary/unbiased-toxic-roberta which outputs a single score:
    - 0.0 = completely safe
    - 1.0 = highly toxic
    
    Args:
        model_name: HuggingFace model identifier
        device: 'cuda', 'cpu', or 'auto'
        batch_size: Number of texts to process at once
        cache_dir: Where to cache the model
    """
    
    def __init__(
        self,
        model_name: str = "unitary/unbiased-toxic-roberta",
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: str = "models/cache"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing ToxicityJudge on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        
        print(f"✓ ToxicityJudge loaded successfully")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        
    def score_text(self, text: str) -> float:
        """
        Score a single text for toxicity.
        
        Args:
            text: Input text to score
            
        Returns:
            Toxicity score between 0.0 (safe) and 1.0 (toxic)
        """
        return self.score_texts([text])[0]
    
    def score_texts(self, texts: List[str]) -> np.ndarray:
        """
        Score multiple texts for toxicity (batched for efficiency).
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of toxicity scores
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get probabilities (softmax over logits)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Extract toxicity score (class 1)
            toxicity_scores = probs[:, 1].cpu().numpy()
        
        # Track inference time
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        avg_time = elapsed / len(texts)
        self.inference_times.append(avg_time)
        
        return toxicity_scores
    
    def score_batch_generator(self, texts: List[str]):
        """
        Score texts in batches (memory efficient for large datasets).
        
        Args:
            texts: List of texts to score
            
        Yields:
            Batches of toxicity scores
        """
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            yield self.score_texts(batch)
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time per text in milliseconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)
    
    def benchmark(self, num_samples: int = 100) -> dict:
        """
        Run a benchmark to measure performance.
        
        Args:
            num_samples: Number of sample texts to test
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"\nRunning benchmark with {num_samples} samples...")
        
        # Generate diverse test samples
        test_texts = [
            "Hello, how are you doing today?",
            "This is great! I love this community.",
            "You are an idiot and should leave.",
            "I completely disagree with your opinion.",
            "Thanks for sharing this information!",
        ] * (num_samples // 5)
        
        # Warm-up run
        _ = self.score_texts(test_texts[:10])
        
        # Clear previous times
        self.inference_times = []
        
        # Benchmark run
        start = time.time()
        scores = self.score_texts(test_texts)
        total_time = (time.time() - start) * 1000  # ms
        
        avg_time = total_time / len(test_texts)
        
        return {
            "num_samples": len(test_texts),
            "total_time_ms": total_time,
            "avg_time_per_text_ms": avg_time,
            "throughput_texts_per_sec": 1000 / avg_time if avg_time > 0 else 0,
            "device": self.device,
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
        }
    
    def save_config(self, path: str):
        """Save judge configuration for reproducibility."""
        import json
        config = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "avg_inference_time_ms": self.get_avg_inference_time(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {path}")


def main():
    """Test the ToxicityJudge with sample texts."""
    
    print("=" * 70)
    print("ToxicityJudge - Unit Test")
    print("=" * 70)
    
    # Initialize judge
    judge = ToxicityJudge()
    
    # Test samples covering the toxicity spectrum
    test_cases = [
        ("Hello world", "benign_greeting"),
        ("This is a great discussion!", "positive_feedback"),
        ("I disagree with your point but respect your view", "respectful_disagreement"),
        ("You are stupid", "mild_insult"),
        ("Everyone here is an idiot", "general_insult"),
        ("You should kill yourself", "severe_toxic"),
    ]
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    
    for text, label in test_cases:
        score = judge.score_text(text)
        
        # Categorize
        if score < 0.3:
            category = "SAFE"
            emoji = "✅"
        elif score < 0.7:
            category = "BORDERLINE"
            emoji = "⚠️"
        else:
            category = "TOXIC"
            emoji = "❌"
        
        print(f"\n{emoji} [{category}] Score: {score:.3f}")
        print(f"   Text: \"{text}\"")
        print(f"   Label: {label}")
    
    # Run benchmark
    print("\n" + "=" * 70)
    print("Performance Benchmark:")
    print("=" * 70)
    
    results = judge.benchmark(num_samples=100)
    
    print(f"\nDevice: {results['device']}")
    print(f"Total samples: {results['num_samples']}")
    print(f"Total time: {results['total_time_ms']:.2f} ms")
    print(f"Average time per text: {results['avg_time_per_text_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_texts_per_sec']:.1f} texts/sec")
    
    # Check performance requirement
    print("\n" + "=" * 70)
    if results['avg_time_per_text_ms'] < 50:
        print(f"✅ PASS: Average inference time ({results['avg_time_per_text_ms']:.2f} ms) < 50 ms")
    else:
        print(f"⚠️  WARNING: Average inference time ({results['avg_time_per_text_ms']:.2f} ms) > 50 ms")
        print("   Consider ONNX optimization (see optimization guide)")
    print("=" * 70)
    
    # Save config
    judge.save_config("configs/toxicity_judge_config.json")
    
    print("\n✓ Day 2 complete!")


if __name__ == "__main__":
    main()
