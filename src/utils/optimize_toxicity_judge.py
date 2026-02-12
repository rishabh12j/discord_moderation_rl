"""
ONNX Optimization for ToxicityJudge
Use this if inference time exceeds 50ms on your hardware.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time

def export_to_onnx(
    model_name: str = "unitary/unbiased-toxic-roberta",
    output_path: str = "models/toxicity_judge.onnx"
):
    """Export model to ONNX format for faster inference."""
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    dummy_text = "This is a test sentence."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
    
    print(f"✓ Model exported to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    return output_path


def benchmark_onnx(onnx_path: str, num_samples: int = 100):
    """Benchmark ONNX model performance."""
    
    print(f"\nBenchmarking ONNX model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unitary/unbiased-toxic-roberta")
    
    # Create ONNX session
    session = ort.InferenceSession(onnx_path)
    
    # Test texts
    test_texts = ["This is a test sentence."] * num_samples
    
    # Warm-up
    for text in test_texts[:10]:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        _ = session.run(None, {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
    
    # Benchmark
    start = time.time()
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        _ = session.run(None, {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
    total_time = (time.time() - start) * 1000
    
    avg_time = total_time / num_samples
    
    print(f"\nONNX Performance:")
    print(f"  Samples: {num_samples}")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Avg time per text: {avg_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} texts/sec")
    
    if avg_time < 50:
        print(f"\n✅ ONNX optimization successful!")
    else:
        print(f"\n⚠️  Still slower than target. Consider CPU optimizations.")


if __name__ == "__main__":
    # Export model
    onnx_path = export_to_onnx()
    
    # Benchmark
    benchmark_onnx(onnx_path)
