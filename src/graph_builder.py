import ast
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta
import hashlib
from typing import List, Dict, Optional, Set, Tuple
import warnings
import joblib

class GraphBuilder:
    """
    Builds graphs from Wazuh logs for GNN link prediction.

    - Training Mode: Groups logs by (filename, type_attack_label, agent_ip)
      and creates graphs with positive (same attack) and negative (inter-graph) edges.
    - Inference Mode: Creates sliding-window graphs grouped by (agent_ip)
      and builds a K-NN graph based on feature similarity.
    
    IMPORTANT:
    You MUST call .fit_scaler(train_df) on your training data *before*
    calling .build_graphs_from_dataframe() for the first time.
    """

    def __init__(
        self,
        output_dir: str,
        numeric_cols: List[str] = ['rule_level', 'rule_firedtimes', 'rule_id'],
        cat_cols: List[str] = ['rule_groups', 'rule_nist_800_53', 'rule_gdpr'],
        ip_cols: List[str] = ['agent_ip', 'data_srcip'],
        desc_col: str = 'description_vector',
        min_nodes_per_graph: int = 10,
        max_nodes_per_graph: int = 1000,
        candidate_edge_topk: int = 10,
        positive_neighbor_window: int = 5,
        time_window_minutes: int = 10,
        neg_edge_ratio: float = 1.0,
        hash_dim_ip: int = 16,
        # Sliding window params for inference
        window_size: int = 500,
        stride: int = 200,
    ):
        self.output_dir = output_dir
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.ip_cols = ip_cols
        self.desc_col = desc_col
        self.scaler = StandardScaler() # Will be fit by fit_scaler()
        self.cat_mappers = {}
        self.candidate_edge_topk = candidate_edge_topk
        self.min_nodes_per_graph = min_nodes_per_graph
        self.max_nodes_per_graph = max_nodes_per_graph
        self.positive_neighbor_window = positive_neighbor_window
        self.time_window_minutes = time_window_minutes
        self.neg_edge_ratio = neg_edge_ratio
        self.hash_dim_ip = hash_dim_ip
        self.window_size = window_size
        self.stride = stride
        os.makedirs(self.output_dir, exist_ok=True)
    
        self._scaler_fitted = False
        self._mappers_fitted = False

    # =========================================================
    # FEATURE PREPROCESSING
    # =========================================================
    
    def _parse_list_cell(self, x) -> list:
        # Handle NumPy arrays directly
        if isinstance(x, np.ndarray):
            return x.tolist()

        # Handle real Python lists
        if isinstance(x, list):
            return x

        # Handle None and NaN
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []

        # Handle string representations
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []

        # Fallback for unexpected types
        return []


    def fit_scaler(self, df: pd.DataFrame):
        """
        Fits the StandardScaler on the provided (training) DataFrame.
        This MUST be called before building any graphs.
        """
        print("Fitting StandardScaler...")
        numeric_data = df[self.numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        if numeric_data.shape[0] > 0:
            self.scaler.fit(numeric_data)
            self._scaler_fitted = True
            print("Scaler fitted successfully.")
        else:
            warnings.warn("No numeric data provided to fit scaler. Scaler will not be used.")
            
    def fit_categorical_mappers(self, df: pd.DataFrame):
        print("Fitting categorical mappers...")
        self.cat_mappers = {}
        df_copy = df.copy()

        for col in self.cat_cols:
            cat_col = df_copy[col].apply(self._parse_list_cell)

            # Explode, drop NaNs, and get ALL unique values
            all_tokens = cat_col.explode().dropna().unique()

            if len(all_tokens) == 0:
                print(f"  - No tokens found for '{col}'. Skipping.")
                self.cat_mappers[col] = {}
                continue

            # Sort them to ensure a consistent, deterministic order
            uniques = sorted(list(all_tokens))
            
            # Store the mapping
            self.cat_mappers[col] = {v: i for i, v in enumerate(uniques)}
            
            print(f"  - For '{col}': found and using {len(uniques)} unique tokens.")
            print(f"  - Tokens: {self.cat_mappers[col].keys()}")
            
        self._mappers_fitted = True
        print("Categorical mappers fitted.")
            
    def _hash_string(self, text: str, dim: int) -> np.ndarray:
        """Return a fixed-length numeric hash embedding for a string."""
        if pd.isna(text) or text in ['None', 'nan', 'none']:
            return np.zeros(dim, dtype=np.float32)

        h = hashlib.md5(str(text).encode()).hexdigest()
        ints = np.array([int(h[i:i+4], 16) for i in range(0, 32, 4)], dtype=np.float32) # 32 chars in md5

        # Repeat or pad to ensure consistent length
        if len(ints) < dim:
            ints = np.tile(ints, int(np.ceil(dim / len(ints))))[:dim]
        else:
            ints = ints[:dim]
        
        # Normalize
        return (ints % 1000) / 1000.0

    def _preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates node features for all nodes in the given DataFrame.
        Assumes scaler has already been fitted.
        """
        if not self._scaler_fitted and self.numeric_cols:
            raise RuntimeError(
                "Scaler has not been fitted. "
            )
        
        if not self._mappers_fitted and self.cat_cols:
            raise RuntimeError(
                "Categorical mappers have not been fitted. "
            )
        
        # 1. Numeric features
        numeric = df[self.numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        numeric_scaled = self.scaler.transform(numeric) if self.numeric_cols else np.zeros((len(df), 0))

        # 2. Categorical features (multi-hot)
        cat_feats_list = []
        for col in self.cat_cols:
            if col not in self.cat_mappers:
                warnings.warn(f"Column '{col}' was not in fitted mappers. Skipping.")
                continue

            mapping = self.cat_mappers[col]
            num_uniques = len(mapping)
            
            if num_uniques == 0:
                continue

            # Use the same robust parser as the fit method
            cat_col = df[col].apply(self._parse_list_cell)
            feat = np.zeros((len(cat_col), num_uniques), dtype=np.float32)
            
            # Iterate over the *list* of tokens, not split-by-comma
            for i, tokens in enumerate(cat_col):
                if not isinstance(tokens, list): continue
                for token in tokens:
                    if token in mapping:
                        feat[i, mapping[token]] = 1.0
            cat_feats_list.append(feat)
        
        cat_feats = np.concatenate(cat_feats_list, axis=1) if cat_feats_list else np.zeros((len(df), 0))

        # 3. Description embedding
        if self.desc_col not in df.columns:
            raise ValueError(f"Description column '{self.desc_col}' not found in DataFrame.")
        
        # Stack the vectors stored in the DataFrame column
        emb = np.stack(df[self.desc_col].values).astype(np.float32)

        # 4. IP features
        ip_feats_list = []
        for col in self.ip_cols:
            feats = np.stack([self._hash_string(v, self.hash_dim_ip) for v in df[col]], axis=0)
            ip_feats_list.append(feats)
        
        ip_feats = np.concatenate(ip_feats_list, axis=1) if ip_feats_list else np.zeros((len(df), 0))

        # Combine all features
        node_feats = np.concatenate([emb, numeric_scaled, cat_feats, ip_feats], axis=1).astype(np.float32)
        return node_feats

    # =========================================================
    # EDGE CREATION
    # =========================================================

    def _create_temporal_edges(
        self, 
        num_nodes: int, 
    ) -> Set[Tuple[int, int]]:
        """
        Creates undirected temporal edges from each node 'i' to the next
        'k' nodes (i+1, ... i+k).
        
        If 'labels' are provided (train mode), it only connects nodes
        if they are NOT benign.
        
        If 'labels' is None (inference mode), connects all temporal neighbors.
        
        Returns a set of (i, j) edge tuples.
        """
        edges: Set[Tuple[int, int]] = set()
        k = self.positive_neighbor_window
        
        for i in range(num_nodes):
            # Connect to the next k neighbors
            for j in range(i + 1, min(i + 1 + k, num_nodes)):
                # Add as sorted tuple to ensure undirected
                edges.add(tuple(sorted((i, j))))
        
        return edges
    

    def _make_edges_training(self, sub_df: pd.DataFrame, full_df: pd.DataFrame):
        """
        Create positive edges (temporal) inside the subgraph and
        negative edges (random) between subgraph and full dataset.
        """
        num_sub_nodes = len(sub_df)

        # === POSITIVE EDGE CREATION ===
        pos_edges_set = self._create_temporal_edges(num_sub_nodes)
        pos_edges = list(pos_edges_set) # Convert set to list for concatenation

        # === NEGATIVE EDGE SAMPLING ===
        # Sample negative nodes from *outside* this subgraph
        # Use indices for quick selection
        sub_df_indices = sub_df.index
        neg_pool = full_df[~full_df.index.isin(sub_df_indices)]
        # remove from neg_pool any log that has the same attack label as sub_df
        sub_attack_labels = set(sub_df['type_attack_label'].unique())
        neg_pool = neg_pool[~neg_pool['type_attack_label'].isin(sub_attack_labels)]
    
        if len(neg_pool) > 0:
            # Sample up to 2x the subgraph size
            n_samples = min(len(neg_pool), num_sub_nodes * 2)
            neg_sample = neg_pool.sample(n_samples, random_state=42)
            # Negative node IDs start *after* the subgraph nodes
            neg_ids = list(range(num_sub_nodes, num_sub_nodes + len(neg_sample)))
        else:
            neg_sample = pd.DataFrame(columns=sub_df.columns)
            neg_ids = []

        # Negative edges: between subgraph nodes (ids) and sampled nodes (neg_ids)
        all_neg_edges = [(i, j) for i in range(num_sub_nodes) for j in neg_ids]
        
        num_pos = len(pos_edges)
        num_neg_to_sample = min(len(all_neg_edges), int(num_pos * self.neg_edge_ratio))
        
        # If no pos edges, sample *some* neg edges anyway for training
        if num_pos == 0 and num_neg_to_sample == 0 and len(all_neg_edges) > 0:
            num_neg_to_sample = min(len(all_neg_edges), num_sub_nodes) # Sample a few
            
        if num_neg_to_sample > 0:
            indices = np.random.choice(len(all_neg_edges), num_neg_to_sample, replace=False)
            sampled_neg_edges = [all_neg_edges[k] for k in indices]
        else:
            sampled_neg_edges = []

        # Combine
        edge_index = np.array(pos_edges + sampled_neg_edges, dtype=np.int64).T
        if edge_index.shape == (0,): # Handle no edges
            edge_index = np.empty((2, 0), dtype=np.int64)
            
        edge_label = np.array([1.0] * len(pos_edges) + [0.0] * len(sampled_neg_edges), dtype=np.float32)

        # Return combined df (for feature generation) and edges
        # We must reset index so that IDs 0..N match the rows
        full_graph_df = pd.concat([sub_df, neg_sample]).reset_index(drop=True)
        
        return edge_index, edge_label, full_graph_df
    

    def _create_knn_edges(self, node_feats: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Creates an undirected K-NN graph based on cosine similarity.
        Returns a set of (i, j) edge tuples.
        """
        if node_feats.shape[0] < 2:
            return set()
            
        sim = cosine_similarity(node_feats)
        np.fill_diagonal(sim, 0.0) # No self-loops

        edges: Set[Tuple[int, int]] = set()
        for i in range(sim.shape[0]):
            # Get indices of top-k most similar nodes
            topk_idx = np.argsort(sim[i])[-self.candidate_edge_topk:]
            
            for j in topk_idx:
                if sim[i, j] > 0.0: # Only add if similarity is positive
                    # Add as sorted tuple to ensure undirected and no duplicates
                    edges.add(tuple(sorted((i, j))))
        return edges

    def _make_edges_inference(self, node_feats: np.ndarray) -> np.ndarray:
        """
        Creates candidate edges for inference by combining:
        1. Temporal Edges (up to 'positive_neighbor_window' neighbors)
        2. K-NN Edges (up to 'candidate_edge_topk' neighbors)
        
        The 'ratio' is controlled by the values of these two __init__ params.
        """
        num_nodes = node_feats.shape[0]
        if num_nodes < 2:
            return np.empty((2, 0), dtype=np.int64)

        # Get temporal edges (undirected)
        temporal_edges = self._create_temporal_edges(num_nodes)
        
        # Get K-NN edges (undirected)
        knn_edges = self._create_knn_edges(node_feats)
        
        # Combine them using set union (automatically handles duplicates)
        combined_edges = temporal_edges.union(knn_edges)
        
        if not combined_edges:
            return np.empty((2, 0), dtype=np.int64)
            
        return np.array(list(combined_edges), dtype=np.int64).T

    # =========================================================
    # GRAPH CREATION
    # =========================================================
    
    def build_graph(
        self, 
        df: pd.DataFrame, 
        graph_id: str, 
        mode: str = "train",
        full_df: Optional[pd.DataFrame] = None
    ):
        metadata_df = None 
        
        if mode == "train":
            if full_df is None:
                raise ValueError("full_df is required for 'train' mode negative sampling.")
            edge_index, edge_label, graph_df = self._make_edges_training(df, full_df)
            node_feats = self._preprocess_features(graph_df)
            metadata_df = graph_df

        elif mode == "inference":
            # Must reset_index so that node 0 is at row 0
            df = df.reset_index(drop=True) 
            node_feats = self._preprocess_features(df)
            edge_index = self._make_edges_inference(node_feats)
            edge_label = None
            metadata_df = df

        else:
            raise ValueError("mode must be 'train' or 'inference'")
            
        # 1. Extract a UNIQUE ID for each NODE (log)
        if 'id' in metadata_df.columns:
            log_ids = metadata_df['id'].astype(str).values
        else:
            warnings.warn("Metadata: No 'id' or 'uuid' found. Using node index as ID.")
            log_ids = np.arange(len(metadata_df)).astype(str)
            
        arrays_dict: Dict[str, np.ndarray] = {
            'node_feats': node_feats, 
            'edge_index': edge_index,
            'log_ids': log_ids,
        }
        
        if edge_label is not None:
            arrays_dict['edge_label'] = edge_label

        out_path = os.path.join(self.output_dir, f"{graph_id}.npz")
        self.save_npz(out_path, arrays_dict) 
        return out_path

    # =========================================================
    # DATAFRAME SPLITTING
    # =========================================================

    def _split_sliding_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Sliding window split for inference mode, respecting max_nodes_per_graph."""
        df = df.sort_values('@timestamp')
        total = len(df)
        subgraphs = []

        if total == 0:
            return []
            
        # If the dataset is small, just return it (capped by max_nodes)
        if total <= self.window_size:
            return [df.iloc[:self.max_nodes_per_graph]]

        for i in range(0, total - self.window_size + 1, self.stride):
            window_df = df.iloc[i:i + self.window_size]

            # If the window is larger than max_nodes_per_graph, split further
            if len(window_df) > self.max_nodes_per_graph:
                for j in range(0, len(window_df), self.max_nodes_per_graph):
                    subgraphs.append(window_df.iloc[j:j + self.max_nodes_per_graph])
            else:
                subgraphs.append(window_df)
        
        # Handle the last partial window if stride doesn't align
        if (total - self.window_size) % self.stride != 0:
             last_window = df.iloc[-self.window_size:]
             if len(last_window) > self.max_nodes_per_graph:
                for j in range(0, len(last_window), self.max_nodes_per_graph):
                    subgraphs.append(last_window.iloc[j:j + self.max_nodes_per_graph])
             else:
                subgraphs.append(last_window)

        return subgraphs

    def _split_by_time(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Time window split for training, respecting max/min node constraints.
        Implements a two-pass approach:
        1. Group by time window.
        2. Merge small chunks forward and fairly split large chunks.
        """
        if len(df) == 0:
            return []
            
        df = df.copy()
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        df = df.sort_values('@timestamp')
        
        start, end = df['@timestamp'].min(), df['@timestamp'].max()
        window = timedelta(minutes=self.time_window_minutes)
        current = start

        # === Group by time window ===
        time_chunks = []
        while current <= end:
            next_t = current + window
            mask = (df['@timestamp'] >= current) & (df['@timestamp'] < next_t)
            chunk = df[mask]
            if not chunk.empty:
                time_chunks.append(chunk)
            current = next_t

        if not time_chunks:
            return []

        # === Merge small chunks and split large chunks ===
        final_subgraphs = []
        buffer = []
        buffer_size = 0

        for i, chunk in enumerate(time_chunks):
            buffer.append(chunk)
            buffer_size += len(chunk)

            is_last_chunk = (i == len(time_chunks) - 1)

            # Process the buffer if it's large enough OR it's the last chunk
            if buffer_size >= self.min_nodes_per_graph or (is_last_chunk and buffer_size > 0):
                
                # Combine all DataFrames in the buffer
                combined_chunk = pd.concat(buffer, ignore_index=True)
                
                # If buffer is too large, split it "fairly"
                if buffer_size > self.max_nodes_per_graph:
                    n_nodes = len(combined_chunk)
                    n_splits = int(np.ceil(n_nodes / self.max_nodes_per_graph))
                    
                    # Calculate a fair split size
                    split_size = int(np.ceil(n_nodes / n_splits))
                    
                    for j in range(0, n_nodes, split_size):
                        sub_chunk = combined_chunk.iloc[j:j + split_size]
                        if not sub_chunk.empty:
                            final_subgraphs.append(sub_chunk)
                
                # If buffer is "just right" (or it's the last leftover)
                else:
                    final_subgraphs.append(combined_chunk)
                
                # Reset the buffer
                buffer = []
                buffer_size = 0

        return final_subgraphs

    # =========================================================
    # MAIN ENTRYPOINT
    # =========================================================
    
    def build_graphs_from_dataframe(
        self, 
        df: pd.DataFrame, 
        description_vectors: np.ndarray, 
        mode: str = "train"
    ) -> List[str]:
        """
        Main entrypoint to build all graphs from a DataFrame.
        
        Args:
            df: The full DataFrame of logs.
            description_vectors: A NumPy array of embeddings,
                                 one for each row in df, in the same order.
            mode: 'train' or 'inference'.
        
        Returns:
            A list of paths to the saved .npz graph files.
        """
        if not self._scaler_fitted and self.numeric_cols:
            raise RuntimeError(
                "Scaler has not been fitted. "
                "Call .fit_scaler(train_df) before building graphs."
            )
            
        if len(df) != len(description_vectors):
            raise ValueError(
                f"DataFrame length ({len(df)}) does not match "
                f"description_vectors length ({len(description_vectors)})."
            )
        
        # Merge embeddings into the DataFrame to ensure they are
        # correctly sliced and sampled with their logs.
        df = df.copy()
        df[self.desc_col] = list(description_vectors)
                
        paths = []

        if mode == "train":
            # Keep original indices for negative sampling
            df = df.reset_index().rename(columns={'index': 'original_index'})
            df = df.set_index('original_index')
            
            # We filter *after* adding embeddings
            df_filtered = df[df["type_attack_label"] != "false_positive"].copy()
            
            # Group by attack-defining properties
            grouped = df_filtered.groupby(["filename", "type_attack_label", "agent_ip"])
            
            print(f"Building {len(grouped)} training groups...")
            for (fname, attack_type, agent), group in tqdm(grouped, desc="Building training graphs"):
                
                # Split group by time windows
                subgraphs = self._split_by_time(group)
                
                for i, sg in enumerate(subgraphs):
                    if len(sg) < 2: continue # Skip graphs with < 2 nodes
                    
                    graph_id = f"train_{fname}_{attack_type}_{agent}_chunk{i}"
                    
                    # Pass the subgraph (sg) and the *full* filtered df (df_filtered)
                    # for negative sampling.
                    paths.append(self.build_graph(
                        sg, 
                        graph_id, 
                        mode, 
                        full_df=df_filtered
                    ))

        elif mode == "inference":
            grouped = df.groupby("agent_ip")
            print(f"Building inference graphs for {len(grouped)} agents...")
            for agent, group in tqdm(grouped, desc="Building inference graphs"):
                
                # Split group by sliding windows
                subgraphs = self._split_sliding_windows(group)
                
                for i, sg in enumerate(subgraphs):
                    if len(sg) < 2: continue
                        
                    graph_id = f"inference_{agent}_win{i}"
                    # No full_df needed for inference
                    paths.append(self.build_graph(
                        sg.reset_index(drop=True), # Reset index for local graph IDs
                        graph_id, 
                        mode, 
                        full_df=None
                    ))

        print(f"Built {len(paths)} graphs in total.")
        return paths

    # =========================================================
    # SAVE
    # =========================================================
    def save_npz(self, out_path: str, arrays_dict: Dict[str, np.ndarray]):
        """Saves a dictionary of arrays to a compressed .npz file."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            np.savez_compressed(out_path, **arrays_dict)
            print(f"Saved graph to {out_path}")
        except Exception as e:
            print(f"Error saving graph to {out_path}: {e}")

    def save_state(self, output_dir: str):
        """Saves the fitted scaler and mappers to disk."""
        if not self._scaler_fitted:
            warnings.warn("Scaler is not fitted. Nothing to save.")
        else:
            scaler_path = os.path.join(output_dir, "graph_builder_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
            
        if not self._mappers_fitted:
            warnings.warn("Mappers are not fitted. Nothing to save.")
        else:
            mappers_path = os.path.join(output_dir, "graph_builder_cat_mappers.json")
            with open(mappers_path, 'w') as f:
                json.dump(self.cat_mappers, f)
            print(f"Saved mappers to {mappers_path}")