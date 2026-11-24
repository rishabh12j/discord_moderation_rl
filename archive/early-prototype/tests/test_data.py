"""
Test data pipeline.
"""

import pandas as pd
from pathlib import Path

def test_raw_data():
    """Test raw data downloaded correctly."""
    file_path = Path("data/raw/jigsaw_toxicity_100k.csv")
    assert file_path.exists(), "Raw data file missing"
    
    df = pd.read_csv(file_path)
    assert len(df) > 50000, "Not enough data"
    assert 'comment_text' in df.columns
    assert 'toxic' in df.columns
    assert 'toxicity_score' in df.columns
    
    print(f"✓ Raw data test passed: {len(df)} messages")


def test_conversations():
    """Test conversation structure."""
    file_path = Path("data/processed/conversations.csv")
    assert file_path.exists(), "Conversations file missing"
    
    df = pd.read_csv(file_path)
    
    # Check columns
    required_cols = ['conversation_id', 'message_id', 'text', 'toxic', 'toxicity_score', 'timestamp']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check conversation structure
    conv_lengths = df.groupby('conversation_id').size()
    assert conv_lengths.min() == 20, "Conversations should have 20 messages"
    assert conv_lengths.max() == 20
    
    print(f"✓ Conversation test passed: {df['conversation_id'].nunique()} conversations")


def test_metadata():
    """Test metadata file."""
    import json
    file_path = Path("data/processed/metadata.json")
    assert file_path.exists(), "Metadata file missing"
    
    with open(file_path) as f:
        meta = json.load(f)
    
    assert 'total_messages' in meta
    assert 'total_conversations' in meta
    assert meta['messages_per_conversation'] == 20
    
    print(f"✓ Metadata test passed")


if __name__ == "__main__":
    print("Running data pipeline tests...\n")
    test_raw_data()
    test_conversations()
    test_metadata()
    print("\n✅ All data tests passed!")





