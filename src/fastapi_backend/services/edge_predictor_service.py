import json
import math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Optional, Tuple
import os

from training_edge_predictor import EdgePredictorGNN


class EdgePredictorService:
    """Service for predicting attack edges using the trained GNN model."""
    
    def __init__(
        self,
        model_path: str = 'models/gnn_edge_predictor_10epochs.pth',
        hidden_channels: int = 128,
    ):
        """
        Initialize the EdgePredictorService.
        
        Args:
            model_path: Path to the trained model weights
            hidden_channels: Number of hidden channels in the model
        """
        self.model_path = model_path
        self.hidden_channels = hidden_channels
        self.model = None
        self.in_channels = None
        
    def load_model(self, in_channels: int):
        """Load the GNN model with specified input channels."""
        if self.model is None or self.in_channels != in_channels:
            print(f"Loading model with {in_channels} input channels...")
            self.model = EdgePredictorGNN(
                in_channels=in_channels,
                hidden_channels=self.hidden_channels
            )
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            self.in_channels = in_channels
            print("Model loaded successfully.")
    
    def eval_graph(self, data: Data) -> torch.Tensor:
        """
        Run the GNN model and return edge probabilities.
        
        Args:
            data: PyG Data object containing node features and edge indices
            
        Returns:
            Tensor of edge probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data)
            probs = torch.sigmoid(logits)
        return probs
    
    def prepare_data(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        log_ids: np.ndarray
    ) -> Data:
        """
        Prepare PyG Data object from numpy arrays.
        
        Args:
            node_features: Node feature matrix (num_nodes, num_features)
            edge_index: Edge connectivity (2, num_edges)
            log_ids: Array of log IDs corresponding to nodes
            
        Returns:
            PyG Data object
        """
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x_tensor, edge_index=edge_index_tensor)
        data.log_ids = log_ids
        return data
    
    def build_attack_graph(
        self,
        probs: torch.Tensor,
        input_data: Data,
        logs_df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Optional[Dict]:
        """
        Build the attack graph from edge probabilities.
        
        Args:
            probs: Edge probabilities from the model
            input_data: Original PyG Data object with log_ids
            logs_df: DataFrame containing log metadata
            threshold: Probability threshold for including edges
            
        Returns:
            Dictionary containing nodes and edges, or None if no edges found
        """
        print(f"Building attack graph with threshold > {threshold}...")
        
        predictions = (probs > threshold).long()
        attack_indices = (predictions == 1).nonzero(as_tuple=True)[0]
        
        if attack_indices.numel() == 0:
            print("No edges exceed the threshold.")
            return None
        
        edges_list = []
        nodes_dict = {}
        max_prob_per_node = defaultdict(float)
        log_ids_array = input_data.log_ids
        
        for idx in attack_indices:
            edge_pos = int(idx.item())
            src_idx = int(input_data.edge_index[0, edge_pos])
            dst_idx = int(input_data.edge_index[1, edge_pos])
            
            src_id = str(log_ids_array[src_idx])
            dst_id = str(log_ids_array[dst_idx])
            prob = float(probs[edge_pos].item())
            
            # Retrieve log details
            try:
                src_row = logs_df.loc[src_id]
                event_time = src_row.get('@timestamp', src_row.get('unix_timestamp', None))
            except KeyError:
                print(f"Warning: Log ID {src_id} or {dst_id} missing from dataset.")
                continue
            
            edges_list.append({
                "timestamp": event_time,
                "source_log_id": src_id,
                "dest_log_id": dst_id,
                "edge_prob": prob
            })
            
            max_prob_per_node[src_id] = max(max_prob_per_node[src_id], prob)
            max_prob_per_node[dst_id] = max(max_prob_per_node[dst_id], prob)
        
        if not edges_list:
            print("No valid edges found for attack graph.")
            return None
        
        # Sort edges temporally
        edges_list = sorted(edges_list, key=lambda e: self._time_key(e))
        
        # Build node metadata
        for node_id, max_prob in max_prob_per_node.items():
            try:
                row = logs_df.loc[str(node_id)]
                nodes_dict[node_id] = {
                    "timestamp": row.get('unix_timestamp', None),
                    "description": row.get('rule_description', None),
                    "rule_id": row.get('rule_id', None),
                    "nist_800_53": row.get('rule_nist_800_53', None),
                    "gdpr": row.get('rule_gdpr', None),
                    "rule_groups": row.get('rule_groups', []),
                    "max_incident_prob": float(max_prob)
                }
            except KeyError:
                nodes_dict[node_id] = {"max_incident_prob": float(max_prob)}
        
        print(f"Attack graph built with {len(nodes_dict)} nodes and {len(edges_list)} edges.")
        return {"nodes": nodes_dict, "edges": edges_list}
    
    @staticmethod
    def _time_key(edge: Dict):
        """Helper to extract sortable timestamp from edge."""
        t = edge.get("timestamp", None)
        if pd.isna(t):
            return pd.Timestamp.max
        if isinstance(t, pd.Timestamp):
            return t
        try:
            return pd.to_datetime(t)
        except Exception:
            return pd.Timestamp.max
    
    def summarize_graph(
        self,
        graph: Dict,
        max_nodes: int = 10,
        max_edges: int = 30
    ) -> Dict:
        """
        Summarize the attack graph for LLM analysis.
        
        Args:
            graph: Attack graph dictionary with nodes and edges
            max_nodes: Maximum number of important nodes to include
            max_edges: Maximum number of important edges to include
            
        Returns:
            Summarized graph dictionary
        """
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])
        
        if not edges or not nodes:
            return {"sample_nodes": [], "sample_edges": []}
        
        # Sort edges and nodes deterministically
        edges = sorted(edges, key=lambda e: (e.get("source_log_id", ""), e.get("dest_log_id", "")))
        node_ids_sorted = sorted(nodes.keys())
        
        # Build directed graph
        G = nx.DiGraph()
        for e in edges:
            src, dst = e["source_log_id"], e["dest_log_id"]
            prob = float(e.get("edge_prob", 0.5))
            G.add_edge(src, dst, weight=-math.log(prob + 1e-6))
        
        # Compute centralities
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        
        # Rank nodes
        scores = []
        for n in node_ids_sorted:
            deg = degree_centrality.get(n, 0)
            bet = betweenness.get(n, 0)
            scores.append((n, (deg + bet) / 2))
        
        ranked_nodes = sorted(scores, key=lambda x: (-x[1], x[0]))
        top_nodes = [n for n, _ in ranked_nodes[:max_nodes]]
        
        # Select top edges
        top_edges = [
            e for e in edges
            if e["source_log_id"] in top_nodes and e["dest_log_id"] in top_nodes
        ]
        top_edges = sorted(
            top_edges,
            key=lambda e: (-e.get("edge_prob", 0.0), e["source_log_id"], e["dest_log_id"])
        )[:max_edges]
        
        # Prepare nodes
        sample_nodes = []
        for nid in sorted(top_nodes):
            node = nodes[nid]
            sample_nodes.append({
                "id": nid,
                "description": node.get("description", ""),
                "rule_groups": node.get("rule_groups", []),
                "rule_id": node.get("rule_id", ""),
                "nist_800_53": node.get("nist_800_53", ""),
                "gdpr": node.get("gdpr", ""),
                "timestamp": str(node.get("timestamp", "")),
                "max_incident_prob": node.get("max_incident_prob", 0.0)
            })
        
        # Prepare edges
        sample_edges = []
        for e in top_edges:
            e_copy = dict(e)
            e_copy["timestamp"] = str(e_copy.get("timestamp", ""))
            sample_edges.append(e_copy)
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "important_nodes": len(sample_nodes),
            "important_edges": len(sample_edges),
            "sample_nodes": sample_nodes,
            "sample_edges": sample_edges
        }
    
    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        log_ids: np.ndarray,
        logs_df: pd.DataFrame,
        threshold: float = 0.5,
        summarize: bool = True
    ) -> Dict:
        """
        Complete prediction pipeline.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            log_ids: Log IDs for nodes
            logs_df: DataFrame with log metadata
            threshold: Probability threshold for edges
            summarize: Whether to return summarized or full graph
            
        Returns:
            Dictionary with prediction results
        """
        # Load model if needed
        in_channels = node_features.shape[1]
        self.load_model(in_channels)
        
        # Prepare data
        data = self.prepare_data(node_features, edge_index, log_ids)
        
        # Get probabilities
        probs = self.eval_graph(data)
        
        # Build attack graph
        graph = self.build_attack_graph(probs, data, logs_df, threshold)
        
        if graph is None:
            return {
                "success": False,
                "message": "No edges exceed the threshold",
                "graph": None
            }
        
        # Optionally summarize
        if summarize:
            graph_data = self.summarize_graph(graph)
        else:
            # Convert to serializable format
            graph_data = {
                "nodes": {k: self._serialize_node(v) for k, v in graph["nodes"].items()},
                "edges": [self._serialize_edge(e) for e in graph["edges"]]
            }
        
        return {
            "success": True,
            "message": f"Found {len(graph['nodes'])} nodes and {len(graph['edges'])} edges",
            "graph": graph_data,
            "threshold": threshold
        }
    
    @staticmethod
    def _serialize_node(node: Dict) -> Dict:
        """Convert node to JSON-serializable format."""
        return {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in node.items()}
    
    @staticmethod
    def _serialize_edge(edge: Dict) -> Dict:
        """Convert edge to JSON-serializable format."""
        return {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in edge.items()}