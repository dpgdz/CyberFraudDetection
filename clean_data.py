"""
Clean dataset by removing problematic samples
"""
import json
from collections import Counter

print("=" * 70)
print("CLEANING DATASET")
print("=" * 70)

def clean_split(input_path, output_path, max_chars=40000):
    """Clean a single split file"""
    
    print(f"\nProcessing: {input_path}")
    print("-" * 70)
    
    samples = []
    removed = {
        'too_long': 0,
        'duplicate': 0,
        'empty': 0
    }
    
    seen_inputs = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            sample = json.loads(line)
            
            # Check for empty fields
            if not sample.get('input', '').strip() or not sample.get('output', '').strip():
                removed['empty'] += 1
                continue
            
            # Check for too long
            if len(sample['input']) > max_chars:
                removed['too_long'] += 1
                print(f"  Removing line {line_num}: {len(sample['input'])} chars (too long)")
                continue
            
            # Check for duplicates
            input_text = sample['input']
            if input_text in seen_inputs:
                removed['duplicate'] += 1
                continue
            
            seen_inputs.add(input_text)
            samples.append(sample)
    
    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  Original: {line_num} samples")
    print(f"  Kept: {len(samples)} samples")
    print(f"  Removed:")
    print(f"    - Too long: {removed['too_long']}")
    print(f"    - Duplicates: {removed['duplicate']}")
    print(f"    - Empty: {removed['empty']}")
    print(f"  Output: {output_path}")
    
    return seen_inputs

# Clean all splits
print("\n1. Cleaning training set...")
train_inputs = clean_split(
    "data/fraud_train_multi_task.jsonl",
    "data/fraud_train_clean.jsonl",
    max_chars=40000
)

print("\n2. Cleaning validation set...")
# Also remove any samples that appear in training
with open("data/fraud_val_multi_task.jsonl", 'r') as f:
    val_samples = []
    overlap_count = 0
    for line in f:
        sample = json.loads(line)
        if sample['input'] not in train_inputs and len(sample['input']) <= 40000:
            val_samples.append(sample)
        else:
            if sample['input'] in train_inputs:
                overlap_count += 1

with open("data/fraud_val_clean.jsonl", 'w') as f:
    for sample in val_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"  Kept: {len(val_samples)} samples")
print(f"  Removed {overlap_count} overlapping with train")
print(f"  Output: data/fraud_val_clean.jsonl")

print("\n3. Cleaning test set...")
# Also remove any samples that appear in training or validation
all_seen = train_inputs | {s['input'] for s in val_samples}

with open("data/fraud_test_multi_task.jsonl", 'r') as f:
    test_samples = []
    overlap_count = 0
    for line in f:
        sample = json.loads(line)
        if sample['input'] not in all_seen and len(sample['input']) <= 40000:
            test_samples.append(sample)
        else:
            if sample['input'] in all_seen:
                overlap_count += 1

with open("data/fraud_test_clean.jsonl", 'w') as f:
    for sample in test_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"  Kept: {len(test_samples)} samples")
print(f"  Removed {overlap_count} overlapping with train/val")
print(f"  Output: data/fraud_test_clean.jsonl")

# Verify cleaned data
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print("\nChecking cleaned data...")
for name, path in [
    ("Train", "data/fraud_train_clean.jsonl"),
    ("Val", "data/fraud_val_clean.jsonl"),
    ("Test", "data/fraud_test_clean.jsonl")
]:
    with open(path, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    max_len = max(len(s['input']) for s in samples)
    avg_len = sum(len(s['input']) for s in samples) / len(samples)
    
    fraud = sum(1 for s in samples if 'classification: fraud' in s['output'].lower())
    legit = len(samples) - fraud
    
    print(f"\n{name}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Max input length: {max_len} chars")
    print(f"  Avg input length: {avg_len:.0f} chars")
    print(f"  Fraud: {fraud} ({fraud/len(samples)*100:.1f}%)")
    print(f"  Legitimate: {legit} ({legit/len(samples)*100:.1f}%)")

print("\n" + "=" * 70)
print("DONE! Use the *_clean.jsonl files for training")
print("=" * 70)
