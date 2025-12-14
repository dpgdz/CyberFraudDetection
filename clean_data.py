import pandas as pd


def clean_csv(input_path: str, output_path: str, max_chars: int = 40000):
    """Clean a single CSV file"""
    
    print(f"\nProcessing: {input_path}")
    print("-" * 70)
    
    # Load CSV
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"  Original samples: {initial_count}")
    
    # Track what gets removed
    removed = {
        'missing': 0,
        'empty': 0,
        'too_long': 0,
        'duplicates': 0
    }
    
    # 1. Remove rows with missing values
    df_clean = df.dropna(subset=['text', 'label', 'domain'])
    removed['missing'] = initial_count - len(df_clean)
    
    # 2. Remove empty text
    df_clean = df_clean[df_clean['text'].str.strip() != '']
    removed['empty'] = initial_count - removed['missing'] - len(df_clean)
    
    # 3. Remove too long text
    text_lengths = df_clean['text'].str.len()
    too_long_mask = text_lengths > max_chars
    removed['too_long'] = too_long_mask.sum()
    
    if removed['too_long'] > 0:
        print(f"\n  Examples of removed long texts:")
        long_samples = df_clean[too_long_mask].head(5)
        for idx, row in long_samples.iterrows():
            text_len = len(row['text'])
            preview = row['text'][:80].replace('\n', ' ')
            print(f"    - {text_len:,} chars: {preview}...")
    
    df_clean = df_clean[~too_long_mask]
    
    # 4. Remove duplicates based on text
    removed['duplicates'] = df_clean.duplicated(subset=['text']).sum()
    df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # Save cleaned CSV
    df_clean.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n  After cleaning: {len(df_clean)} samples")
    print(f"  Removed:")
    print(f"    - Missing/NaN: {removed['missing']}")
    print(f"    - Empty text: {removed['empty']}")
    print(f"    - Too long (>{max_chars:,} chars): {removed['too_long']}")
    print(f"    - Duplicates: {removed['duplicates']}")
    print(f"    - Total removed: {sum(removed.values())} ({sum(removed.values())/initial_count*100:.2f}%)")
    
    # Check cleaned data stats
    max_len = df_clean['text'].str.len().max()
    avg_len = df_clean['text'].str.len().mean()
    
    print(f"\n  Cleaned data stats:")
    print(f"    - Max text length: {max_len:,} chars")
    print(f"    - Avg text length: {avg_len:,.0f} chars")
    
    # Label distribution
    label_counts = df_clean['label'].value_counts()
    print(f"\n  Label distribution:")
    for label, count in label_counts.items():
        print(f"    - {label}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    print(f"\n  Output: {output_path}")
    
    return df_clean


def remove_cross_split_duplicates(train_path, val_path, test_path):
    """Remove samples that appear in multiple splits"""
    
    print("\n" + "=" * 70)
    print("REMOVING CROSS-SPLIT DUPLICATES (DATA LEAKAGE)")
    print("=" * 70)
    
    # Load all splits
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\nBefore removing overlaps:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Get train texts
    train_texts = set(train_df['text'].values)
    
    # Remove from val if in train
    val_overlap = val_df['text'].isin(train_texts)
    val_overlap_count = val_overlap.sum()
    val_df_clean = val_df[~val_overlap]
    
    # Remove from test if in train or val
    all_seen_texts = train_texts | set(val_df_clean['text'].values)
    test_overlap = test_df['text'].isin(all_seen_texts)
    test_overlap_count = test_overlap.sum()
    test_df_clean = test_df[~test_overlap]
    
    # Save if changes were made
    if val_overlap_count > 0:
        val_df_clean.to_csv(val_path, index=False)
        print(f"\n  Removed {val_overlap_count} samples from validation (overlap with train)")
    else:
        print(f"\n  No overlaps found in validation")
    
    if test_overlap_count > 0:
        test_df_clean.to_csv(test_path, index=False)
        print(f"  Removed {test_overlap_count} samples from test (overlap with train/val)")
    else:
        print(f"  No overlaps found in test")
    
    print(f"\nAfter removing overlaps:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df_clean)} samples")
    print(f"  Test:  {len(test_df_clean)} samples")
    
    return len(train_df), len(val_df_clean), len(test_df_clean)


if _name_ == "_main_":
    print("=" * 70)
    print("CSV DATA CLEANING")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Remove samples with text >40,000 characters")
    print("  2. Remove duplicate samples")
    print("  3. Remove empty/missing values")
    print("  4. Remove cross-split duplicates (data leakage)")
    
    # Define file paths
    splits = [
        ("data/train.csv", "data/train_clean.csv"),
        ("data/validation.csv", "data/validation_clean.csv"),
        ("data/test.csv", "data/test_clean.csv"),
    ]
    
    # Clean each split
    for input_path, output_path in splits:
        print("\n" + "=" * 70)
        clean_csv(input_path, output_path, max_chars=40000)
    
    # Remove cross-split duplicates
    train_count, val_count, test_count = remove_cross_split_duplicates(
        "data/train_clean.csv",
        "data/validation_clean.csv",
        "data/test_clean.csv"
    )
