"""
Test user feature engineering.
"""

import pandas as pd
import json
from pathlib import Path

def test_user_features_file():
    """Test user features CSV."""
    file_path = Path("data/processed/user_features.csv")
    assert file_path.exists(), "User features file missing"
    
    df = pd.read_csv(file_path)
    
    # Check columns
    required_cols = ['user_id', 'total_messages', 'toxic_messages', 
                     'avg_toxicity', 'user_profile', 'join_days_ago']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check profiles
    assert set(df['user_profile'].unique()) == {'good_user', 'borderline', 'troll'}
    
    # Check toxicity ordering
    good_tox = df[df['user_profile'] == 'good_user']['avg_toxicity'].mean()
    border_tox = df[df['user_profile'] == 'borderline']['avg_toxicity'].mean()
    troll_tox = df[df['user_profile'] == 'troll']['avg_toxicity'].mean()
    
    assert good_tox < border_tox < troll_tox, "Toxicity ordering incorrect"
    
    print(f"✓ User features test passed: {len(df)} users")
    print(f"  Good users avg toxicity: {good_tox:.3f}")
    print(f"  Borderline avg toxicity: {border_tox:.3f}")
    print(f"  Trolls avg toxicity: {troll_tox:.3f}")


def test_user_lookup():
    """Test user lookup JSON."""
    file_path = Path("data/processed/user_lookup.json")
    assert file_path.exists(), "User lookup file missing"
    
    with open(file_path) as f:
        lookup = json.load(f)
    
    # Check structure
    assert len(lookup) > 1000, "Not enough users in lookup"
    
    # Check first user has required keys
    first_user = list(lookup.values())[0]
    required_keys = ['avg_toxicity', 'total_messages', 'toxic_messages', 
                     'profile', 'join_days_ago']
    for key in required_keys:
        assert key in first_user, f"Missing key: {key}"
    
    print(f"✓ User lookup test passed: {len(lookup)} users")


def test_conversations_with_users():
    """Test updated conversations file."""
    file_path = Path("data/processed/conversations_with_users.csv")
    assert file_path.exists(), "Conversations with users file missing"
    
    df = pd.read_csv(file_path)
    
    # Check new columns
    assert 'user_id' in df.columns
    assert 'user_profile' in df.columns
    
    # Check no nulls
    assert df['user_id'].notna().all(), "Null user_ids found"
    
    print(f"✓ Conversations with users test passed: {len(df)} messages")


if __name__ == "__main__":
    print("Running user feature tests...\n")
    test_user_features_file()
    test_user_lookup()
    test_conversations_with_users()
    print("\n✅ All user feature tests passed!")
