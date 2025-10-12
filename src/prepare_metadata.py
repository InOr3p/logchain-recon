import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# --- CONFIGURATION ---
GRAPH_DIR = os.path.join("out", "graphs")
METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

# --- HELPER FUNCTION TO FIX LABELS ---
def get_correct_label(filename_part, label_part):
    """
    Reconstructs the full attack label from parts incorrectly split by underscores.
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

# --- SCRIPT ---
print(f"Scanning directory: {GRAPH_DIR}")
graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.npz')]
print(f"Found {len(graph_files)} graph files.")

if not graph_files:
    raise FileNotFoundError("No .npz graph files found in the specified directory!")

metadata = []
for npz_file in tqdm(graph_files, desc="Parsing filenames"):
    # Filename format: {filename}_{attack_label}_{agent_ip}_session{session_idx}.npz
    base_name = npz_file.replace('.npz', '')
    
    # rsplit from the right is still the best approach to isolate the last 3 components
    parts = base_name.rsplit('_', 3)
    
    if len(parts) == 4:
        filename_part, parsed_label, agent_ip, session_str = parts
        
        # Apply the correction logic to reconstruct the full label
        final_label = get_correct_label(filename_part, parsed_label)
        
        metadata.append({
            'filepath': os.path.join(GRAPH_DIR, npz_file),
            'label_name': final_label
        })
    else:
        print(f"Warning: Could not parse filename '{npz_file}'. Skipping.")

df_meta = pd.DataFrame(metadata)

# --- Create a vocabulary for labels ---
# This maps each string label (e.g., 'dirb') to an integer (e.g., 2)
unique_labels = sorted(df_meta['label_name'].unique()) # Sort for consistent mapping
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df_meta['label_id'] = df_meta['label_name'].map(label_to_id)

# Save the label mapping for later use
label_map_path = os.path.join(METADATA_DIR, 'label_map.json')
with open(label_map_path, 'w') as f:
    json.dump(label_to_id, f, indent=4)

print("\n--- Corrected Label Distribution ---")
print(df_meta['label_name'].value_counts())
print(f"\nCreated label mapping and saved to {label_map_path}")

# --- Split data into train, validation, and test sets ---
train_val_df, test_df = train_test_split(
    df_meta,
    test_size=0.15,
    random_state=42,
    stratify=df_meta['label_id'] # Ensures same label distribution in splits
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15, # 15% of the remaining 85%
    random_state=42,
    stratify=train_val_df['label_id']
)

# Save the splits
train_df.to_csv(os.path.join(METADATA_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(METADATA_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(METADATA_DIR, 'test.csv'), index=False)

print("\n--- Data Split Summary ---")
print(f"Training set size:   {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size:       {len(test_df)}")
print(f"\nMetadata preparation complete. Files saved in '{METADATA_DIR}' directory.")