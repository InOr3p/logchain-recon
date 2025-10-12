import pandas as pd
import os
import numpy as np
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph_builder import GraphBuilder
import logging
from tqdm import tqdm


# === CONFIGURATION ===
extract_dir = "extracted_dataset"
output_dir = os.path.join("out", "graphs")
os.makedirs(output_dir, exist_ok=True)

max_logs_threshold = 5000       # maximum logs per session
session_window_minutes = 10     # time window for splitting large groups
timestamp_col = "@timestamp"    # timestamp column name
num_threads = 4                 # number of concurrent threads

# === LOGGING CONFIG ===
log_format = "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("build_graphs_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === LOAD DATA ===
logger.info("Loading dataset...")
df = pd.read_parquet(os.path.join(extract_dir, 'sorted_ds_with_labels.parquet'))
description_vectors = np.load(os.path.join(extract_dir, 'vectorized_descr.npy'))
df['description_vector'] = list(description_vectors)
logger.info(f"Dataset shape: {df.shape}")

# Ensure timestamp is datetime
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
df = df.dropna(subset=[timestamp_col])

# === GROUP DATA ===
grouped = df.groupby(['filename', 'attack_label', 'agent_ip'])
logger.info(f"Total groups: {len(grouped)}")

# Show number of logs per group
group_counts = grouped.size().reset_index(name='num_logs')
logger.info("Logs per group (top 10 largest):")
logger.info(f"\n{group_counts.sort_values('num_logs', ascending=False).head(10)}")

# === GRAPH BUILDER ===
gb = GraphBuilder()

# === FUNCTION TO SPLIT LARGE GROUPS ===
def split_large_group_by_time_threshold(group_df, max_logs, window_minutes):
    group_df = group_df.sort_values(timestamp_col).reset_index(drop=True)
    chunks = []

    start_idx = 0
    current_start_time = group_df.loc[start_idx, timestamp_col]
    current_end_time = current_start_time + timedelta(minutes=window_minutes)

    for i in range(len(group_df)):
        # Stop current chunk if threshold or time window exceeded
        if (group_df.loc[i, timestamp_col] > current_end_time) or (i - start_idx >= max_logs):
            chunk = group_df.iloc[start_idx:i]
            chunks.append(chunk)

            # Start new chunk
            start_idx = i
            current_start_time = group_df.loc[start_idx, timestamp_col]
            current_end_time = current_start_time + timedelta(minutes=window_minutes)

    # Add last remaining chunk
    if start_idx < len(group_df):
        chunks.append(group_df.iloc[start_idx:])

    return chunks

# === THREAD WORKER ===
def process_subgroup_thread(args):
    filename, attack_label, agent_ip, session_idx, sub_df = args

    output_path = os.path.join(
        output_dir,
        f"{filename}_{attack_label}_{agent_ip}_session{session_idx}.npz"
    )

    # Skip if graph already exists
    if os.path.exists(output_path):
        logger.info(f"Skipping existing graph {output_path}")
        return output_path

    try:
        logger.info(f"Building graph for {filename}-{attack_label}-{agent_ip} (session {session_idx}) | logs={len(sub_df)}")
        G = gb.build_from_df(sub_df)
        gb.save_npz(output_path, G)
        return output_path
    except Exception as e:
        logger.exception(f"Error building graph for {filename}-{attack_label}-{agent_ip} session{session_idx}")
        return None

# === PREPARE TASKS ===
tasks = []
for (filename, attack_label, agent_ip), group_df in grouped:
    # Split large groups
    if len(group_df) > max_logs_threshold:
        subgroups = split_large_group_by_time_threshold(group_df, max_logs_threshold, session_window_minutes)
        logger.info(f"Group {filename}-{attack_label}-{agent_ip} split into {len(subgroups)} subgroups")
    else:
        subgroups = [group_df]

    # Add tasks for threads
    for session_idx, sub_df in enumerate(subgroups, 1):
        tasks.append((filename, attack_label, agent_ip, session_idx, sub_df))

logger.info(f"--- Total tasks to process: {len(tasks)} ---\n")

# === RUN THREADS ===
results = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_subgroup_thread, t) for t in tasks]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Building graphs", ncols=90):
        try:
            result = future.result()
            if result is not None:
                results.append(result)
        except Exception as e:
            logging.error(f"Exception in thread: {e}")

logger.info(f"\nFinished building {len(results)} graphs.")
