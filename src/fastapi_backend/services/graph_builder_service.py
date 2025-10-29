import ast
from collections import Counter
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta
import hashlib
from typing import List, Dict, Optional, Set, Tuple
import warnings
import joblib
import json

class GraphBuilder:
    """
    Builds graphs from Wazuh logs for GNN link prediction.
    
    - INFERENCE MODE ONLY -
    
    Creates sliding-window graphs grouped by (agent_ip)
    and builds a K-NN graph based on feature similarity.
    
    IMPORTANT:
    You MUST call .load_state(path) to load the pre-trained
    scaler and mappers before using this class.
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
        hash_dim_ip: int = 16,
        # Sliding window params
        window_size: int = 500,
        stride: int = 200,
    ):
        self.output_dir = output_dir
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.ip_cols = ip_cols
        self.desc_col = desc_col
        self.candidate_edge_topk = candidate_edge_topk
        self.min_nodes_per_graph = min_nodes_per_graph
        self.max_nodes_per_graph = max_nodes_per_graph
        self.positive_neighbor_window = positive_neighbor_window
        self.hash_dim_ip = hash_dim_ip
        self.window_size = window_size
        self.stride = stride
        os.makedirs(self.output_dir, exist_ok=True)
        
        # These will be loaded by load_state()
        self.scaler = None
        self.cat_mappers = {}
        self._scaler_fitted = False
        self._mappers_fitted = False

    # =========================================================
    # STATE LOADING
    # =========================================================
    
    def load_state(self, input_dir: str):
        """Loads a pre-fitted scaler and mappers from disk."""
        try:
            scaler_path = os.path.join(input_dir, "graph_builder_scaler.joblib")
            self.scaler = joblib.load(scaler_path)
            self._scaler_fitted = True
            print(f"Loaded scaler from {scaler_path}")
        except FileNotFoundError:
            raise RuntimeError(f"Could not find scaler at {scaler_path}")
            
        try:
            mappers_path = os.path.join(input_dir, "graph_builder_cat_mappers.json")
            with open(mappers_path, 'r') as f:
                self.cat_mappers = json.load(f)
            self._mappers_fitted = True
            print(f"Loaded mappers from {mappers_path}")
        except FileNotFoundError:
            raise RuntimeError(f"Could not find mappers at {mappers_path}")
            
    # =========================================================
    # FEATURE PREPROCESSING (Kept from original)
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

    def _hash_string(self, text: str, dim: int) -> np.ndarray:
        if pd.isna(text) or text in ['None', 'nan', 'none']:
            return np.zeros(dim, dtype=np.float32)
        h = hashlib.md5(str(text).encode()).hexdigest()
        ints = np.array([int(h[i:i+4], 16) for i in range(0, 32, 4)], dtype=np.float32)
        if len(ints) < dim:
            ints = np.tile(ints, int(np.ceil(dim / len(ints))))[:dim]
        else:
            ints = ints[:dim]
        return (ints % 1000) / 1000.0

    def _preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates node features for all nodes in the given DataFrame.
        Assumes scaler and mappers have already been loaded.
        """
        if not self._scaler_fitted:
            raise RuntimeError("Scaler has not been loaded. Call .load_state() first.")
        if not self._mappers_fitted:
            raise RuntimeError("Mappers have not been loaded. Call .load_state() first.")
        
        # 1. Numeric features
        numeric = df[self.numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        numeric_scaled = self.scaler.transform(numeric) if self.numeric_cols else np.zeros((len(df), 0))

        # 2. Categorical features (multi-hot)
        cat_feats_list = []
        for col in self.cat_cols:
            if col not in self.cat_mappers:
                warnings.warn(f"Column '{col}' was not in loaded mappers. Skipping.")
                continue
            mapping = self.cat_mappers[col]
            num_uniques = len(mapping)
            if num_uniques == 0:
                continue

            cat_col = df[col].apply(self._parse_list_cell)
            feat = np.zeros((len(cat_col), num_uniques), dtype=np.float32)
            
            for i, tokens in enumerate(cat_col):
                if not isinstance(tokens, list): continue
                for token in tokens:
                    if token in mapping:
                        feat[i, mapping[token]] = 1.0
            cat_feats_list.append(feat)
        
        cat_feats = np.concatenate(cat_feats_list, axis=1) if cat_feats_list else np.zeros((len(df), 0))

        # 3. Description embedding
        if self.desc_col not in df.columns:
            raise ValueError(f"Description column '{self.desc_col}' not found.")
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

    # Create temporal edges (edges between logs close in time)
    def _create_temporal_edges(self, num_nodes: int) -> Set[Tuple[int, int]]:
        """Creates undirected temporal edges from 'i' to the next 'k' nodes."""
        edges: Set[Tuple[int, int]] = set()
        k = self.positive_neighbor_window
        for i in range(num_nodes):
            for j in range(i + 1, min(i + 1 + k, num_nodes)):
                edges.add(tuple(sorted((i, j))))
        return edges

    # Create K-NN edges based on cosine similarity between node features
    def _create_knn_edges(self, node_feats: np.ndarray) -> Set[Tuple[int, int]]:
        """Creates an undirected K-NN graph based on cosine similarity."""
        if node_feats.shape[0] < 2:
            return set()
        sim = cosine_similarity(node_feats)
        np.fill_diagonal(sim, 0.0)
        edges: Set[Tuple[int, int]] = set()
        for i in range(sim.shape[0]):
            topk_idx = np.argsort(sim[i])[-self.candidate_edge_topk:]
            for j in topk_idx:
                if sim[i, j] > 0.0:
                    edges.add(tuple(sorted((i, j))))
        return edges

    def _make_edges_inference(self, node_feats: np.ndarray) -> np.ndarray:
        """Creates candidate edges for inference."""
        num_nodes = node_feats.shape[0]
        if num_nodes < 2:
            return np.empty((2, 0), dtype=np.int64)
            
        temporal_edges = self._create_temporal_edges(num_nodes)
        knn_edges = self._create_knn_edges(node_feats)
        combined_edges = temporal_edges.union(knn_edges)
        
        if not combined_edges:
            return np.empty((2, 0), dtype=np.int64)
        return np.array(list(combined_edges), dtype=np.int64).T

    # =========================================================
    # GRAPH CREATION
    # =========================================================
    
    def build_graph(self, df: pd.DataFrame, graph_id: str):
        """Builds a single inference graph."""
        
        # Must reset_index so that node 0 is at row 0
        df = df.reset_index(drop=True) 
        node_feats = self._preprocess_features(df)
        edge_index = self._make_edges_inference(node_feats)
        
        if 'id' in df.columns:
            log_ids = df['id'].astype(str).values
        else:
            warnings.warn("Metadata: No 'id' found. Using node index as ID.")
            log_ids = np.arange(len(df)).astype(str)
            
        arrays_dict: Dict[str, np.ndarray] = {
            'node_feats': node_feats, 
            'edge_index': edge_index,
            'log_ids': log_ids,
        }
        
        out_path = os.path.join(self.output_dir, f"{graph_id}.npz")
        self.save_npz(out_path, arrays_dict) 
        return out_path

    # =========================================================
    # DATAFRAME SPLITTING (Inference method only)
    # =========================================================

    def _split_sliding_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Sliding window split for inference mode."""
        df = df.sort_values('timestamp')
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

    # =========================================================
    # MAIN ENTRYPOINT (Simplified for inference)
    # =========================================================
    
    def build_graphs_from_dataframe(
        self, 
        df: pd.DataFrame, 
        description_vectors: np.ndarray, 
    ) -> List[str]:
        """
        Main entrypoint to build all inference graphs from a DataFrame.
        """
        if not self._scaler_fitted or not self._mappers_fitted:
            raise RuntimeError(
                "Scaler/Mappers have not been loaded. "
                "Call .load_state(path) before building graphs."
            )
        if len(df) != len(description_vectors):
            raise ValueError(
                f"DataFrame length ({len(df)}) does not match "
                f"description_vectors length ({len(description_vectors)})."
            )
        
        df = df.copy()
        df[self.desc_col] = list(description_vectors)
        paths = []

        grouped = df.groupby("agent_ip")
        print(f"Building inference graphs for {len(grouped)} agents...")
        for agent, group in tqdm(grouped, desc="Building inference graphs"):
            
            subgraphs = self._split_sliding_windows(group)
            
            for i, sg in enumerate(subgraphs):
                if len(sg) < 2: continue
                
                graph_id = f"inference_{agent}_win{i}"
                paths.append(self.build_graph(
                    sg.reset_index(drop=True), 
                    graph_id,
                ))

        print(f"Built {len(paths)} graphs in total.")
        return paths
    

    def save_npz(self, out_path: str, arrays_dict: Dict[str, np.ndarray]):
        """Saves a dictionary of arrays to a compressed .npz file."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            np.savez_compressed(out_path, **arrays_dict)
            print(f"Saved graph to {out_path}")
        except Exception as e:
            print(f"Error saving graph to {out_path}: {e}")