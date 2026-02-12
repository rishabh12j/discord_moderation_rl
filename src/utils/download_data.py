"""
Download and prepare toxicity dataset for Discord moderation RL.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import json

def download_jigsaw_data(output_dir="data/raw"):
    """
    Download Jigsaw Toxic Comment dataset.
    
    Returns preprocessed DataFrame with:
    - comment_text: message content
    - toxic: binary label (0=safe, 1=toxic)
    - toxicity_score: continuous score 0-1
    """
    
    print("Downloading Jigsaw Toxic Comment dataset...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load from HuggingFace
    dataset = load_dataset("google/civil_comments", split="train[:100000]")  # First 100k for speed
    
    # Convert to pandas
    df = pd.DataFrame(dataset)
    
    # Create unified toxicity score
    df['toxicity_score'] = df['toxicity']  # Already 0-1 continuous
    df['toxic'] = (df['toxicity'] > 0.5).astype(int)  # Binary label
    
    # Rename columns
    df = df.rename(columns={'text': 'comment_text'})
    
    # Keep only essential columns
    df = df[['comment_text', 'toxic', 'toxicity_score']]
    
    # Remove very short/long messages
    df = df[df['comment_text'].str.len().between(10, 500)]
    
    # Remove nulls
    df = df.dropna()
    
    # Save
    output_file = output_path / "jigsaw_toxicity_100k.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Downloaded {len(df)} messages to {output_file}")
    print(f"  Toxic: {df['toxic'].sum()} ({df['toxic'].mean()*100:.1f}%)")
    print(f"  Safe: {(~df['toxic']).sum()} ({(~df['toxic']).mean()*100:.1f}%)")
    
    return df


def create_synthetic_conversations(df, output_dir="data/processed", 
                                   conversation_length=20, num_conversations=5000):
    """
    Group messages into synthetic conversation threads.
    
    Simulates Discord channel conversations with mixed toxic/safe messages.
    """
    
    print(f"\nCreating {num_conversations} synthetic conversations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    conversations = []
    
    # Separate toxic and safe messages
    toxic_df = df[df['toxic'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    safe_df = df[df['toxic'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
    
    toxic_idx = 0
    safe_idx = 0
    
    for conv_id in range(num_conversations):
        messages = []
        
        # Randomly decide conversation toxicity profile
        # 70% mostly safe, 20% mixed, 10% mostly toxic
        import random
        profile = random.choices(['safe', 'mixed', 'toxic'], weights=[0.7, 0.2, 0.1])[0]
        
        for msg_idx in range(conversation_length):
            # Decide if this message should be toxic
            if profile == 'safe':
                use_toxic = random.random() < 0.05  # 5% toxic
            elif profile == 'mixed':
                use_toxic = random.random() < 0.3   # 30% toxic
            else:  # toxic
                use_toxic = random.random() < 0.7   # 70% toxic
            
            # Get message
            if use_toxic and toxic_idx < len(toxic_df):
                row = toxic_df.iloc[toxic_idx]
                toxic_idx += 1
            elif safe_idx < len(safe_df):
                row = safe_df.iloc[safe_idx]
                safe_idx += 1
            else:
                break  # Ran out of messages
            
            messages.append({
                'conversation_id': conv_id,
                'message_id': msg_idx,
                'text': row['comment_text'],
                'toxic': int(row['toxic']),
                'toxicity_score': float(row['toxicity_score']),
                'timestamp': msg_idx,  # Sequential for now
            })
        
        if len(messages) == conversation_length:
            conversations.extend(messages)
    
    # Convert to DataFrame
    conv_df = pd.DataFrame(conversations)
    
    # Save
    output_file = output_path / "conversations.csv"
    conv_df.to_csv(output_file, index=False)
    
    print(f"✓ Created {num_conversations} conversations ({len(conv_df)} total messages)")
    print(f"  Saved to {output_file}")
    
    # Statistics
    print("\nConversation statistics:")
    print(f"  Average toxicity per conversation: {conv_df.groupby('conversation_id')['toxic'].mean().mean():.2%}")
    print(f"  Conversations with >50% toxic: {(conv_df.groupby('conversation_id')['toxic'].mean() > 0.5).sum()}")
    
    return conv_df


def main():
    print("=" * 70)
    print("Day 3: Data Ingestion Pipeline")
    print("=" * 70)
    
    # Download raw data
    df = download_jigsaw_data()
    
    # Create conversation structure
    conv_df = create_synthetic_conversations(df)
    
    # Save metadata
    metadata = {
        'total_messages': len(df),
        'total_conversations': conv_df['conversation_id'].nunique(),
        'messages_per_conversation': 20,
        'toxicity_rate': float(df['toxic'].mean()),
        'dataset': 'jigsaw_civil_comments',
        'date': pd.Timestamp.now().isoformat(),
    }
    
    with open('data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Day 3 Complete!")
    print("=" * 70)
    print("\nNext: Day 4 - User History Feature Engineering")


if __name__ == "__main__":
    main()
