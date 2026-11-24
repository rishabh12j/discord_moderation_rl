"""
Compare toxicity detection models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.toxicity_judge import ToxicityJudge
import time

# Test texts
test_texts = [
    "I love this community!",
    "You're an idiot.",
    "Can someone help?",
    "Get lost, loser.",
    "Great discussion!",
    "Shut up, moron.",
] * 10  # 60 texts

print("Comparing toxicity models...\n")

# Test XLM-RoBERTa (new)
print("1. XLM-RoBERTa Large (textdetox)")
judge_xlmr = ToxicityJudge(
    model_name="textdetox/xlmr-large-toxicity-classifier",
    device="cuda",
    batch_size=16
)

start = time.time()
results_xlmr = judge_xlmr.predict(test_texts)
time_xlmr = time.time() - start

toxic_count_xlmr = results_xlmr['toxic'].sum()
print(f"   Time: {time_xlmr:.3f}s")
print(f"   Throughput: {len(test_texts)/time_xlmr:.1f} texts/sec")
print(f"   Toxic detected: {toxic_count_xlmr}/{len(test_texts)}")

print("\n" + "="*50)
print("✅ Model comparison complete!")
