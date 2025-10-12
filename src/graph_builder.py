import threading
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import os
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
import logging

# --------- Helpers ----------
def make_event_id(prefix, idx):
    return f"{prefix}-{idx:08d}"

def node_key(node_type, key):
    return f"{node_type}:{key}"


# --------- Main builder ----------
class GraphBuilder:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8,
                 max_time_diff: int = 180):
        """
        model_name: SentenceTransformer model to use.
        similarity_threshold: cosine similarity threshold to create an edge.
        max_time_diff: maximum allowed time difference (in seconds) to connect logs temporally.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Loading GraphBuilder instance...")
        self.embedder = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_time_diff = max_time_diff

    def build_from_df(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed graph from a dataframe of logs.

        Each log becomes a node, and temporal/similarity-based edges
        are added between logs that occur within a certain time window
        and have similar semantic embeddings.
        """
        thread_name = threading.current_thread().name
        self.logger.info(f"[{thread_name}] Starting graph build on {len(df)} logs")

        # === Data Cleaning ===
        # Remove logs missing timestamp or description
        df = df.dropna(subset=['@timestamp', 'rule_description']).reset_index(drop=True)
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])

        # Identify if the group contains a single log
        is_single_log = len(df) == 1

        # === Embedding Generation ===
        # If the dataframe doesn't have precomputed vectors, encode descriptions
        if 'description_vector' not in df.columns:
            self.logger.info(f"[{thread_name}] Computing embeddings using SentenceTransformer...")
            embeddings = self.embedder.encode(
                df['rule_description'].tolist(),
                normalize_embeddings=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                show_progress_bar=False
            )
        else:
            # Stack precomputed embeddings into a NumPy array
            embeddings = np.vstack(df['description_vector'].values)
            embeddings = normalize(embeddings)  # ensure unit-norm for cosine similarity

        # === Sanity Checks ===
        # Ensure embeddings match the number of logs and are 2D
        if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
            self.logger.warning(f"[{thread_name}] Invalid embedding shape {embeddings.shape} for {len(df)} logs — skipping graph")
            return nx.DiGraph(single_log_group=is_single_log)

        # Handle NaN values safely
        if np.any(np.isnan(embeddings)):
            self.logger.warning(f"[{thread_name}] NaN values detected in embeddings — replacing with zeros")
            embeddings = np.nan_to_num(embeddings)

        # === Initialize Graph ===
        G = nx.DiGraph(single_log_group=is_single_log)

        # Add one node per log entry
        for i, row in df.iterrows():
            node_id = f"{row['agent_ip']}_{i}"
            G.add_node(
                node_id,
                timestamp=row['@timestamp'],
                agent_ip=row['agent_ip'],
                description=row['rule_description'],
                embedding=embeddings[i],
                is_single_log=is_single_log
            )

        # === Handle Single-Log Case ===
        if is_single_log:
            self.logger.info(f"[{thread_name}] Single-log group detected: {len(G.nodes)} node, {len(G.edges)} edges")
            return G  # No edges to create, return early

        # === Prepare Data for Edge Creation ===
        df_sorted = df.sort_values(by='@timestamp').reset_index(drop=True)
        node_ids = [f"{df_sorted.loc[i, 'agent_ip']}_{df_sorted.index[i]}" for i in range(len(df_sorted))]

        self.logger.info(f"[{thread_name}] Building edges using vectorized similarity...")

        # === Build Edges (Time + Similarity Based) ===
        for i in range(len(df_sorted) - 1):
            t_i = df_sorted.loc[i, '@timestamp']

            # Compute time differences to all following logs
            time_diffs = (df_sorted['@timestamp'][i + 1:] - t_i).dt.total_seconds().to_numpy()

            # Identify valid neighbors within the time window
            valid_mask = time_diffs <= self.max_time_diff
            if not np.any(valid_mask):
                continue  # No neighbors in time window

            valid_j = np.where(valid_mask)[0] + (i + 1)  # shift to match df_sorted indices

            # Compute cosine similarity between current node and valid neighbors
            sims = embeddings[i] @ embeddings[valid_j].T

            # === Edge Creation ===
            for idx, j in enumerate(valid_j):
                if sims[idx] >= self.similarity_threshold:
                    try:
                        G.add_edge(
                            node_ids[i],
                            node_ids[j],
                            weight=float(sims[idx]),
                            time_diff=float(time_diffs[j - (i + 1)])
                        )
                    except Exception as edge_err:
                        # Log full context for debugging if edge creation fails
                        self.logger.exception(
                            f"[{thread_name}] Failed to add edge "
                            f"i={i}, j={j}, idx={idx}, len(time_diffs)={len(time_diffs)}, "
                            f"sims_shape={sims.shape}, embeddings_shape={embeddings.shape}"
                        )

        # === Wrap Up ===
        self.logger.info(f"[{thread_name}] Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G


    def graph_to_arrays(self, G: nx.DiGraph):
        """Convert NetworkX DiGraph to TGN/GraphSAGE arrays."""
        node2id = {node: idx for idx, node in enumerate(G.nodes())}
        sources, destinations, timestamps, edge_feats = [], [], [], []
        node_feats = []

        for node in G.nodes():
            node_feats.append(G.nodes[node]['embedding'])

        for u, v, data in G.edges(data=True):
            sources.append(node2id[u])
            destinations.append(node2id[v])
            timestamps.append(pd.Timestamp(G.nodes[v]['timestamp']).timestamp())
            edge_feats.append([data['weight']])

        arrays_dict = {
            'sources': np.array(sources, dtype=np.int64),
            'destinations': np.array(destinations, dtype=np.int64),
            'timestamps': np.array(timestamps, dtype=np.float32),
            'edge_feats': np.array(edge_feats, dtype=np.float32),
            'node_feats': np.array(node_feats, dtype=np.float32),
            'single_log_group': np.array([int(G.graph.get('single_log_group', False))])
        }
        return arrays_dict

    def save_npz(self, out_path: str, G: nx.DiGraph):
        """Convert graph to arrays and save safely as .npz"""
        if os.path.exists(out_path):
            self.logger.info(f"Graph already exists at {out_path}, skipping.")
            return

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        arrays_dict = self.graph_to_arrays(G)
        if len(arrays_dict['node_feats']) == 0:
            raise ValueError(f"Graph {out_path} has no nodes!")

        np.savez_compressed(out_path, **arrays_dict)
        tag = " (single-log group)" if arrays_dict['single_log_group'][0] == 1 else ""
        self.logger.info(f"Saved graph to {out_path}{tag} with keys: {list(arrays_dict.keys())}")
