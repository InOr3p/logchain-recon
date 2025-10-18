import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta


class GraphBuilder:
    """
    Scalable Graph Builder for large log datasets (millions of rows).

    Supports:
    - 'train' mode: builds labeled subgraphs for GNN link prediction.
    - 'inference' mode: builds candidate subgraphs from new unlabeled logs.

    Automatically splits large datasets into subgraphs (time-based or random sampling)
    to keep each graph small enough for GNN training.
    """

    def __init__(
        self,
        output_dir: str,
        description_vectors: np.ndarray,
        numeric_cols: List[str] = ['rule_level', 'rule_firedtimes'],
        cat_cols: List[str] = ['rule_groups', 'rule_nist_800_53', 'rule_gdpr'],
        max_nodes_per_graph: int = 1000,
        candidate_edge_topk: int = 10,
        time_window_minutes: int = 10,
        neg_edge_ratio: float = 1.0,  # ratio of negatives to positives
    ):
        self.output_dir = output_dir
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()
        self.candidate_edge_topk = candidate_edge_topk
        self.description_vectors = description_vectors
        self.max_nodes_per_graph = max_nodes_per_graph
        self.time_window_minutes = time_window_minutes
        self.neg_edge_ratio = neg_edge_ratio
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================================
    # ---------------- FEATURE CONSTRUCTION -------------------
    # =========================================================
    def _preprocess_features(self, df: pd.DataFrame):
        """Build numeric + categorical + embedding features."""
        numeric = df[self.numeric_cols].fillna(0).to_numpy()
        numeric_scaled = self.scaler.fit_transform(numeric)

        # categorical one-hot encoding
        cat_feats = []
        for col in self.cat_cols:
            cat_col = df[col].fillna('none').astype(str)
            uniques = list(set(v for val in cat_col for v in val.split(',') if v))
            uniques = uniques[:16]  # cap to reduce dimensionality
            mapping = {v: i for i, v in enumerate(uniques)}
            feat = np.zeros((len(cat_col), len(uniques)), dtype=np.float32)
            for i, val in enumerate(cat_col):
                for token in val.split(','):
                    if token in mapping:
                        feat[i, mapping[token]] = 1.0
            cat_feats.append(feat)

        if cat_feats:
            cat_feats = np.concatenate(cat_feats, axis=1)
        else:
            cat_feats = np.zeros((len(df), 0))

        # description embeddings
        emb = self.description_vectors[:len(df)]
        node_feats = np.concatenate([emb, numeric_scaled, cat_feats], axis=1).astype(np.float32)
        return node_feats

    # =========================================================
    # ---------------- EDGE CREATION (TRAIN) -------------------
    # =========================================================
    def _make_edges_training(self, df: pd.DataFrame):
        """
        Create positive/negative edges with balanced sampling.
        Positive = same attack_label, Negative = different.
        """
        ids = list(range(len(df)))
        labels = df['attack_label'].to_numpy()
        pos_edges = [(i, j) for i, j in combinations(ids, 2) if labels[i] == labels[j] and labels[i] != 'benign']
        
        # sample negatives to keep balance
        all_pairs = list(combinations(ids, 2))
        neg_candidates = [(i, j) for (i, j) in all_pairs if labels[i] != labels[j]]
        if len(pos_edges) > 0:
            num_neg = min(len(neg_candidates), int(len(pos_edges) * self.neg_edge_ratio))
            neg_edges = np.random.choice(len(neg_candidates), num_neg, replace=False)
            neg_edges = [neg_candidates[k] for k in neg_edges]
        else:
            neg_edges = []

        edge_index = np.array(pos_edges + neg_edges, dtype=np.int64).T
        edge_label = np.array([1] * len(pos_edges) + [0] * len(neg_edges), dtype=np.float32)
        return edge_index, edge_label

    # =========================================================
    # ---------------- EDGE CREATION (INFERENCE) ---------------
    # =========================================================
    def _make_edges_inference(self, node_feats: np.ndarray):
        """Weak edge generator using cosine similarity (top-k)."""
        sim = cosine_similarity(node_feats)
        np.fill_diagonal(sim, 0.0)

        edges = []
        for i in range(sim.shape[0]):
            topk_idx = np.argsort(sim[i])[-self.candidate_edge_topk:]
            for j in topk_idx:
                if sim[i, j] > 0:
                    edges.append((i, j))

        return np.array(edges, dtype=np.int64).T

    # =========================================================
    # ------------------- GRAPH BUILDING -----------------------
    # =========================================================
    def build_graph(self, df: pd.DataFrame, graph_id: str, mode: str = "train"):
        node_feats = self._preprocess_features(df)
        if mode == "train":
            edge_index, edge_label = self._make_edges_training(df)
        elif mode == "inference":
            edge_index = self._make_edges_inference(node_feats)
            edge_label = None
        else:
            raise ValueError("mode must be 'train' or 'inference'")

        arrays_dict = {'node_feats': node_feats, 'edge_index': edge_index}
        if edge_label is not None:
            arrays_dict['edge_label'] = edge_label

        out_path = os.path.join(self.output_dir, f"{graph_id}.npz")
        self.save_npz(out_path, arrays_dict)
        return out_path

    # =========================================================
    # ------------------- GRAPH SPLITTING ----------------------
    # =========================================================
    def _split_by_agent_and_time(self, df: pd.DataFrame):
        """Split logs per agent_ip and within time windows."""
        df['@timestamp'] = pd.to_datetime(df['@timestamp'], errors='coerce')
        valid_df = df.dropna(subset=['@timestamp'])
        
        subgraphs = []
        for agent, group in valid_df.groupby('agent_ip'):
            group = group.sort_values('@timestamp')
            start, end = group['@timestamp'].min(), group['@timestamp'].max()
            window = timedelta(minutes=self.time_window_minutes)

            current = start
            while current < end:
                next_t = current + window
                mask = (group['@timestamp'] >= current) & (group['@timestamp'] < next_t)
                chunk = group[mask]
                if len(chunk) >= 5:
                    subgraphs.append(chunk)
                current = next_t

            # include leftover logs not covered by windows
            covered_idx = pd.concat(subgraphs).index if subgraphs else []
            leftovers = group.drop(index=covered_idx, errors='ignore')
            if not leftovers.empty:
                subgraphs.append(leftovers)
        return subgraphs


    def _split_large_attack(self, df: pd.DataFrame):
        """
        If attack has too many logs, randomly sample chunks to keep graphs manageable.
        """
        chunks = []
        total = len(df)
        indices = np.arange(total)
        np.random.shuffle(indices)
        for i in range(0, total, self.max_nodes_per_graph):
            chunk_idx = indices[i:i+self.max_nodes_per_graph]
            chunks.append(df.iloc[chunk_idx])
        return chunks

    # =========================================================
    # ------------------- MAIN ENTRYPOINT ----------------------
    # =========================================================
    def build_graphs_from_dataframe(self, df: pd.DataFrame, group_col: str = "type_attack_label", mode: str = "train"):
        """
        Build a collection of graphs (training or inference).
        - For training: builds multiple subgraphs per attack.
        - For inference: builds one large candidate graph.
        """
        paths = []

        if mode == "train":
            assert group_col in df.columns, f"{group_col} column missing for training mode"
            grouped = df.groupby(group_col)

            for label, group in tqdm(grouped, desc=f"Building training graphs"):
                # further split large attacks by time or sampling
                if len(group) > self.max_nodes_per_graph:
                    subgraphs = self._split_by_agent_and_time(group)
                    if not subgraphs:  # fallback if timestamps not usable
                        subgraphs = self._split_large_attack(group)
                else:
                    subgraphs = [group]

                for i, sg in enumerate(subgraphs):
                    graph_id = f"{label}_chunk{i}"
                    path = self.build_graph(sg.reset_index(drop=True), graph_id, mode)
                    paths.append(path)
        else:
            # In inference mode: one big graph of recent logs
            if len(df) > self.max_nodes_per_graph:
                df = df.tail(self.max_nodes_per_graph)  # keep latest logs
            graph_id = "inference_graph"
            path = self.build_graph(df.reset_index(drop=True), graph_id, mode)
            paths.append(path)

        return paths

    # =========================================================
    # ------------------- SAVE GRAPH ---------------------------
    # =========================================================
    def save_npz(self, out_path: str, arrays_dict: dict):
        if os.path.exists(out_path):
            print(f"Graph already exists at {out_path}, skipping.")
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **arrays_dict)
        print(f"Saved graph to {out_path} with keys: {list(arrays_dict.keys())}")
