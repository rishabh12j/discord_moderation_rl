"""
User History Feature Engineering
Calculates user-level statistics and behavioral profiles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

class UserFeatureExtractor:
    """
    Extract and compute user-level features from conversation data.
    
    Creates:
    - User toxicity baselines (average toxicity per user)
    - User behavior profiles (good_user, troll, new_user, etc.)
    - User history statistics
    """
    
    def __init__(self, conversations_path="data/processed/conversations.csv"):
        """Load conversation data."""
        self.conv_df = pd.read_csv(conversations_path)
        print(f"Loaded {len(self.conv_df)} messages from {self.conv_df['conversation_id'].nunique()} conversations")
        
        # Create synthetic user IDs (in real Discord, these would be actual user IDs)
        self._assign_synthetic_users()
        
    def _assign_synthetic_users(self):
        """
        Assign synthetic user IDs to messages.
        
        Simulates Discord users with different behavioral patterns:
        - 70% good users (mostly safe messages)
        - 20% borderline users (mixed)
        - 10% trolls (mostly toxic)
        """
        print("\nAssigning synthetic user profiles...")
        
        user_ids = []
        user_profiles = []
        
        # Group by conversation
        for conv_id, group in self.conv_df.groupby('conversation_id'):
            messages = group.sort_values('message_id')
            
            for idx, row in messages.iterrows():
                # Assign user based on message toxicity
                if row['toxic'] == 1:
                    # Toxic message - likely from troll or borderline user
                    if np.random.random() < 0.7:
                        profile = 'troll'
                        user_id = f"troll_{np.random.randint(0, 500)}"
                    else:
                        profile = 'borderline'
                        user_id = f"borderline_{np.random.randint(0, 1000)}"
                else:
                    # Safe message - likely from good user or borderline
                    if np.random.random() < 0.85:
                        profile = 'good_user'
                        user_id = f"good_{np.random.randint(0, 3000)}"
                    else:
                        profile = 'borderline'
                        user_id = f"borderline_{np.random.randint(0, 1000)}"
                
                user_ids.append(user_id)
                user_profiles.append(profile)
        
        self.conv_df['user_id'] = user_ids
        self.conv_df['user_profile'] = user_profiles
        
        print(f"✓ Assigned {self.conv_df['user_id'].nunique()} unique users")
        print(f"  Good users: {(self.conv_df['user_profile'] == 'good_user').sum()}")
        print(f"  Borderline users: {(self.conv_df['user_profile'] == 'borderline').sum()}")
        print(f"  Trolls: {(self.conv_df['user_profile'] == 'troll').sum()}")
    
    def compute_user_statistics(self):
        """
        Compute per-user statistics.
        
        Returns:
            DataFrame with columns:
            - user_id
            - total_messages
            - toxic_messages
            - avg_toxicity
            - user_profile
            - join_days_ago (simulated)
        """
        print("\nComputing user statistics...")
        
        user_stats = []
        
        for user_id, user_messages in self.conv_df.groupby('user_id'):
            # Basic counts
            total_messages = len(user_messages)
            toxic_messages = user_messages['toxic'].sum()
            avg_toxicity = user_messages['toxicity_score'].mean()
            
            # User profile
            profile = user_messages['user_profile'].iloc[0]
            
            # Simulate join date (days ago)
            # Good users: 30-365 days
            # Borderline: 7-90 days
            # Trolls: 0-30 days
            if profile == 'good_user':
                join_days_ago = np.random.randint(30, 365)
            elif profile == 'borderline':
                join_days_ago = np.random.randint(7, 90)
            else:  # troll
                join_days_ago = np.random.randint(0, 30)
            
            user_stats.append({
                'user_id': user_id,
                'total_messages': total_messages,
                'toxic_messages': toxic_messages,
                'avg_toxicity': avg_toxicity,
                'user_profile': profile,
                'join_days_ago': join_days_ago,
            })
        
        user_df = pd.DataFrame(user_stats)
        
        print(f"✓ Computed statistics for {len(user_df)} users")
        print(f"\nUser statistics by profile:")
        print(user_df.groupby('user_profile')['avg_toxicity'].describe())
        
        return user_df
    
    def create_user_lookup(self, user_df):
        """
        Create fast lookup dictionary for RL environment.
        
        Returns:
            dict: {user_id -> {features}}
        """
        print("\nCreating user lookup table...")
        
        user_lookup = {}
        
        for _, row in user_df.iterrows():
            user_lookup[row['user_id']] = {
                'avg_toxicity': float(row['avg_toxicity']),
                'total_messages': int(row['total_messages']),
                'toxic_messages': int(row['toxic_messages']),
                'profile': row['user_profile'],
                'join_days_ago': int(row['join_days_ago']),
            }
        
        print(f"✓ Created lookup table with {len(user_lookup)} users")
        
        return user_lookup
    
    def save_features(self, user_df, user_lookup, output_dir="data/processed"):
        """Save user features to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        user_df.to_csv(output_path / "user_features.csv", index=False)
        print(f"✓ Saved user features to {output_path / 'user_features.csv'}")
        
        # Save lookup dict
        with open(output_path / "user_lookup.json", 'w') as f:
            json.dump(user_lookup, f, indent=2)
        print(f"✓ Saved user lookup to {output_path / 'user_lookup.json'}")
        
        # Update conversations with user IDs
        self.conv_df.to_csv(output_path / "conversations_with_users.csv", index=False)
        print(f"✓ Saved updated conversations to {output_path / 'conversations_with_users.csv'}")
        
        return user_df, user_lookup


def main():
    print("=" * 70)
    print("Day 4: User History Feature Engineering")
    print("=" * 70)
    
    # Initialize extractor
    extractor = UserFeatureExtractor()
    
    # Compute user statistics
    user_df = extractor.compute_user_statistics()
    
    # Create lookup table
    user_lookup = extractor.create_user_lookup(user_df)
    
    # Save everything
    extractor.save_features(user_df, user_lookup)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    print(f"\nTotal unique users: {len(user_df)}")
    print(f"Average messages per user: {user_df['total_messages'].mean():.1f}")
    print(f"Average user toxicity: {user_df['avg_toxicity'].mean():.3f}")
    
    print("\nToxicity by user profile:")
    for profile in ['good_user', 'borderline', 'troll']:
        profile_df = user_df[user_df['user_profile'] == profile]
        print(f"  {profile}: {profile_df['avg_toxicity'].mean():.3f} "
              f"(n={len(profile_df)}, messages={profile_df['total_messages'].sum()})")
    
    print("\n" + "=" * 70)
    print("✅ Day 4 Complete!")
    print("=" * 70)
    print("\nNext: Day 5 - Pre-compute Embeddings")


if __name__ == "__main__":
    main()
