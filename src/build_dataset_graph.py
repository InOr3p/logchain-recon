import os
import numpy as np
import pandas as pd
from graph_builder import GraphBuilder

# ----------------------------
# CONFIGURATION
# ----------------------------

EXTRACT_DIR = "extracted_dataset"
OUTPUT_DIR = "graphs_dataset"         # directory where graphs will be saved
MODE = "train"                        # "train" or "inference"
MIN_NODES_PER_GRAPH = 5
MAX_NODES_PER_GRAPH = 1000
TIME_WINDOW_MINUTES = 10

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
