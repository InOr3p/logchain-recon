import os
import numpy as np
import pandas as pd
from graph_builder import GraphBuilder
import re
from collections import Counter, defaultdict
import random
import shutil

# ----------------------------
# CONFIGURATION
# ----------------------------

EXTRACT_DIR = "extracted_dataset"
OUTPUT_DIR = "graphs_dataset"         # directory where graphs will be saved
BALANCED_DIR = "graphs_dataset_balanced"
MODE = "train"                        # "train" or "inference"
MIN_NODES_PER_GRAPH = 5
MAX_NODES_PER_GRAPH = 1000
TIME_WINDOW_MINUTES = 10
CAP_PER_CLASS = 200
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def build_graphs_from_df():
    # ----------------------------
    # LOAD DATA
    # ----------------------------

    print("\nLoading dataset...")
    logs_path = os.path.join(EXTRACT_DIR, "sorted_ds_with_labels.parquet")
    embeddings_path = os.path.join(EXTRACT_DIR, "vectorized_descr.npy")

    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Missing file: {logs_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Missing file: {embeddings_path}")

    logs_df = pd.read_parquet(logs_path)
    description_vectors = np.load(embeddings_path)

    print(f"Loaded {len(logs_df):,} logs and {description_vectors.shape[0]:,} embeddings")

    # ----------------------------
    # CLEAN AND PREPARE DATA
    # ----------------------------

    # ensure timestamps are datetime
    logs_df['@timestamp'] = pd.to_datetime(logs_df['@timestamp'], errors='coerce')
    logs_df = logs_df.dropna(subset=['@timestamp'])

    logs_df['data_srcip'] = logs_df['data_srcip'].fillna('none')

    print("Dataset cleaned and ready.")

    # ----------------------------
    # INSTANTIATE GRAPH BUILDER
    # ----------------------------

    builder = GraphBuilder(
        output_dir=OUTPUT_DIR,
        min_nodes_per_graph=MIN_NODES_PER_GRAPH,
        max_nodes_per_graph=MAX_NODES_PER_GRAPH,
        time_window_minutes=TIME_WINDOW_MINUTES,
    )

    print("Fitting scaler on training data...")
    builder.fit_scaler(logs_df)

    print("Fitting categorical mappers on training data...")
    builder.fit_categorical_mappers(logs_df)

    builder.save_state(EXTRACT_DIR)

    # ----------------------------
    # BUILD GRAPHS
    # ----------------------------

    print(f"\nStarting graph building in {MODE.upper()} mode...")

    graph_paths = builder.build_graphs_from_dataframe(
        df=logs_df,
        description_vectors=description_vectors,
        mode="train",
    )

    print(f"\nFinished building graphs! {len(graph_paths)} graphs saved to: {OUTPUT_DIR}")
    for p in graph_paths[:5]:
        print(f" - {p}")
    if len(graph_paths) > 5:
        print(f" ...and {len(graph_paths)-5} more.")


def analyze_graph_distribution(directory: str):
    """
    Scans the graph directory, parses filenames, and counts
    the number of graphs generated for each attack type.
    """
    print(f"Analyzing graph distribution in: {directory}\n")
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return

    filename_regex = re.compile(r"train_.*?_(.*?)_\d.*_chunk\d+\.npz")

    attack_labels = []
    
    for filename in os.listdir(directory):
        match = filename_regex.match(filename)
        
        if match:
            # The attack_type is the first (and only) captured group
            attack_type = match.group(1)
            attack_labels.append(attack_type)

    if not attack_labels:
        print("No 'train' graphs found with the expected format.")
        print("Expected format: train_{fname}_{attack_type}_{agent}_chunk{i}.npz")
        return

    # Count the occurrences of each label
    label_counts = Counter(attack_labels)
    
    print("--- Graph Count per Attack Type ---")
    
    # Sort by the most common (highest count) first
    total_graphs = 0
    for label, count in label_counts.most_common():
        print(f"  - {label:<25} {count:>7,} graphs")
        total_graphs += count
        
    print("-----------------------------------")
    print(f"  Total {total_graphs:,} training graphs found.\n")

    # --- Print Imbalance Warning ---
    if label_counts:
        most_common_count = label_counts.most_common(1)[0][1]
        least_common_count = label_counts.most_common()[-1][1]
        
        if most_common_count > least_common_count * 50:
            print("  WARNING: Your dataset is severely imbalanced.")
            print(f"    The largest class has {most_common_count:,} graphs,")
            print(f"    while the smallest has {least_common_count:,} graphs.")
            print("\n    Consider subsampling the larger classes to create a")
            print("    balanced training set, as discussed.")


def create_balanced_graphs_dataset():
    """
    Creates a new directory with a "capped" number of graphs per class
    by subsampling the majority classes and keeping all of the minority classes.
    """
    print(f"Source directory: {OUTPUT_DIR}")
    print(f"Target directory: {BALANCED_DIR}")
    print(f"Cap per class: {CAP_PER_CLASS}\n")
    
    if os.path.exists(BALANCED_DIR):
        print(f"Warning: Target directory '{BALANCED_DIR}' already exists. Deleting it...")
        shutil.rmtree(BALANCED_DIR)
    
    os.makedirs(BALANCED_DIR)

    # Discover all graph files and group them by class
    filename_regex = re.compile(r"train_.*?_(.*?)_\d.*_chunk\d+\.npz")
    files_by_class = defaultdict(list)
    
    print("Scanning source directory...")
    for filename in os.listdir(OUTPUT_DIR):
        match = filename_regex.match(filename)
        if match:
            attack_type = match.group(1)
            files_by_class[attack_type].append(filename)

    if not files_by_class:
        print("Error: No graph files found in source directory. Exiting.")
        return

    # Apply capped subsampling
    total_files_copied = 0
    print("Applying capped subsampling and copying files...")
    
    for attack_type, files in files_by_class.items():
        files_to_copy = []
        
        if len(files) > CAP_PER_CLASS:
            # Subsample
            files_to_copy = random.sample(files, CAP_PER_CLASS)
            print(f"  - {attack_type:<25} Subsampled {len(files)} -> {len(files_to_copy)}")
        else:
            # Keep all
            files_to_copy = files
            print(f"  - {attack_type:<25} Kept all {len(files_to_copy)}")
        
        # Copy the selected files to the new directory
        for filename in files_to_copy:
            src_path = os.path.join(OUTPUT_DIR, filename)
            dst_path = os.path.join(BALANCED_DIR, filename)
            shutil.copy(src_path, dst_path)
            total_files_copied += 1

    print("\n-----------------------------------")
    print(f"Created balanced dataset with {total_files_copied} graphs.")
    print(f"New dataset location: {BALANCED_DIR}")


    
def main():
    build_graphs_from_df()
    create_balanced_graphs_dataset()
    analyze_graph_distribution(BALANCED_DIR)

if __name__ == "__main__":
    main()