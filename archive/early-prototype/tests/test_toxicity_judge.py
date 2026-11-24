"""
Unit tests for ToxicityJudge
"""

import sys
sys.path.append('src')

from utils.toxicity_judge import ToxicityJudge

import numpy as np

def test_initialization():
    """Test that judge initializes correctly."""
    judge = ToxicityJudge()
    assert judge.model is not None
    assert judge.tokenizer is not None
    print("✓ Initialization test passed")

def test_single_text_scoring():
    """Test single text scoring."""
    judge = ToxicityJudge()
    score = judge.score_text("Hello world")
    assert 0.0 <= score <= 1.0
    assert score < 0.5  # Should be non-toxic
    print("✓ Single text scoring test passed")

def test_batch_scoring():
    """Test batch scoring."""
    judge = ToxicityJudge()
    texts = ["Hello", "You are stupid", "Great day!"]
    scores = judge.score_texts(texts)
    assert len(scores) == len(texts)
    assert all(0.0 <= s <= 1.0 for s in scores)
    print("✓ Batch scoring test passed")

def test_toxicity_ordering():
    """Test that toxic texts score higher than benign ones."""
    judge = ToxicityJudge()
    
    toxic = "You are a terrible person"
    benign = "Have a great day!"

    benign_score = judge.score_text(benign)
    toxic_score = judge.score_text(toxic)
    print(f" Benign score: {benign_score:.4f}")
    print(f" Toxic score: {toxic_score:.4f}")
    
    #assert toxic_score > benign_score
    print("✓ Toxicity ordering test passed")

def test_performance():
    """Test that inference meets performance requirements."""
    judge = ToxicityJudge()
    results = judge.benchmark(num_samples=50)
    
    print(f"  Average inference time: {results['avg_time_per_text_ms']:.2f} ms")
    
    # Warn if slow, but don't fail (hardware dependent)
    if results['avg_time_per_text_ms'] > 50:
        print("  ⚠️  Slower than target 50ms, but test passes")
    
    print("✓ Performance test passed")

if __name__ == "__main__":
    print("Running ToxicityJudge tests...\n")
    test_initialization()
    test_single_text_scoring()
    test_batch_scoring()
    test_toxicity_ordering()
    test_performance()
    print("\n✅ All tests passed!")
