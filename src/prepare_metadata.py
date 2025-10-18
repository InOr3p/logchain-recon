import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from collections import Counter

# CONFIGURATION
GRAPH_DIR = os.path.join("out", "graphs")
METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

# HELPER FUNCTION: FIX LABELS
def get_correct_label(filename_part, label_part):
    """
    Reconstructs the full attack label from parts incorrectly split by underscores.
    Example: 'wheeler_false_positive_...' -> 'false_positive'
    """
    if label_part == 'positive' and filename_part.endswith('_false'):
        return 'false_positive'
    if label_part == 'scans' and filename_part.endswith('_network'):
        return 'network_scans'
    if label_part == 'scans' and filename_part.endswith('_service'):
        return 'service_scans'
    if label_part == 'shell' and filename_part.endswith('_reverse'):
        return 'reverse_shell'
    if label_part == 'escalation' and filename_part.endswith('_privilege'):
        return 'privilege_escalation'
    if label_part == 'stop' and filename_part.endswith('_service'):
        return 'service_stop'
    return label_part


# --- SCAN GRAPH DIRECTORY ---
print(f"Scanning directory: {GRAPH_DIR}")
graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.npz')]
print(f"Found {len(graph_files)} graph files.\n")

metadata = []
for npz_file in tqdm(graph_files, desc="Parsing filenames"):
    # Expected filename format:
    # {filename}_{attack_label}_{agent_ip}_session{session_idx}.npz
    base_name = npz_file.replace('.npz', '')
    parts = base_name.rsplit('_', 3)

    if len(parts) == 4:
        filename_part, parsed_label, agent_ip, session_str = parts
        final_label = get_correct_label(filename_part, parsed_label)
        metadata.append({
            'filepath': os.path.join(GRAPH_DIR, npz_file),
            'label_name': final_label
        })
    else:
        print(f"Warning: Could not parse filename '{npz_file}'. Skipping.")

df_meta = pd.DataFrame(metadata)
print("\n--- Raw Dataset Distribution ---")
print(df_meta['label_name'].value_counts())

# --- REMOVE CLASSES TOO SMALL TO LEARN ---
UNLEARNABLE_THRESHOLD = 3  # Classes with <3 samples are removed
value_counts = df_meta['label_name'].value_counts()
to_remove = value_counts[value_counts < UNLEARNABLE_THRESHOLD].index.tolist()

if to_remove:
    print(f"\nRemoving unlearnable classes: {to_remove}")
    df_meta = df_meta[~df_meta['label_name'].isin(to_remove)]

print("\n--- Cleaned Label Distribution ---")
print(df_meta['label_name'].value_counts())

# --- ENCODE LABELS ---
unique_labels = sorted(df_meta['label_name'].unique())
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df_meta['label_id'] = df_meta['label_name'].map(label_to_id)

# Save label map for use in training/evaluation
with open(os.path.join(METADATA_DIR, 'label_map.json'), 'w') as f:
    json.dump(label_to_id, f, indent=4)

print("\nEncoded labels and saved label_map.json")


# --- TRAIN / VAL / TEST SPLIT (STRATIFIED) ---
train_val_df, test_df = train_test_split(
    df_meta,
    test_size=0.15,
    random_state=42,
    stratify=df_meta['label_id']
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15,
    random_state=42,
    stratify=train_val_df['label_id']
)

print("\n--- Split Summary ---")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# --- COMPUTE CLASS WEIGHTS FOR LOSS & SAMPLING ---
label_counts = Counter(train_df['label_id'])
total = sum(label_counts.values())

# Inverse frequency weighting
class_weights = {label: total / count for label, count in label_counts.items()}
# Normalize (optional)
norm_factor = sum(class_weights.values()) / len(class_weights)
class_weights = {k: v / norm_factor for k, v in class_weights.items()}

# Save for later use in training
with open(os.path.join(METADATA_DIR, 'class_weights.json'), 'w') as f:
    json.dump(class_weights, f, indent=4)

print("\nSaved class_weights.json for training-time balancing.")

train_df.to_csv(os.path.join(METADATA_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(METADATA_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(METADATA_DIR, 'test.csv'), index=False)

print(f"\nAll metadata files saved to '{METADATA_DIR}' directory.")
print("\n--- Final Label Distribution (Train Split) ---")
print(train_df['label_name'].value_counts())
