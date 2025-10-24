import json
import os
import math
import numpy as np
import pandas as pd
import requests
import torch
from collections import defaultdict
from torch_geometric.data import Data
import networkx as nx
from training_edge_predictor import EdgePredictorGNN
import random

random.seed(42)
np.random.seed(42)

EXTRACT_DIR = 'extracted_dataset'
HIDDEN_CHANNELS = 128

# ===========================================================
# --------------------- HELPER FUNCTIONS --------------------
# ===========================================================

def eval_graph(model, data, device='cpu'):
    """Run the GNN model and return edge probabilities."""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits)
    return probs


def load_dataset():
    print("\nLoading dataset...")
    logs_path = os.path.join(EXTRACT_DIR, "sorted_ds_with_labels.parquet")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Missing file: {logs_path}")

    logs_df = pd.read_parquet(logs_path)
    logs_df['id'] = logs_df['id'].astype(str)
    logs_df = logs_df.set_index('id')
    logs_df['@timestamp'] = pd.to_datetime(logs_df['@timestamp'], errors='coerce')
    return logs_df


def load_model(in_channels):
    print("Instantiating model structure...")
    model = EdgePredictorGNN(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS)

    model_path = os.path.join('models', 'gnn_edge_predictor_10epochs.pth')
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    print("Model weights loaded successfully.")
    return model


def load_graph():
    graph_path = os.path.join('graphs_dataset_balanced', 'train_fox_cracking_172.17.130.196_chunk0.npz')
    print(f"Loading graph from {graph_path}...")
    
    loaded_data = np.load(graph_path, allow_pickle=True)
    print(f"Keys found in .npz file: {loaded_data.files}")

    node_features_np = loaded_data['node_feats']
    edge_index_np = loaded_data['edge_index']
    log_ids_np = loaded_data['log_ids']

    in_channels = node_features_np.shape[1]
    print(f"\nSuccessfully detected {in_channels} input features (in_channels).")

    return node_features_np, edge_index_np, log_ids_np, in_channels


def prepare_data(node_features_np, edge_index_np, log_ids_np):
    x_tensor = torch.tensor(node_features_np, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    data.log_ids = log_ids_np
    print(f"\nSuccessfully created PyG Data object:\n{data}")
    return data


# ===========================================================
# ---------------- BUILD ATTACK GRAPH -----------------------
# ===========================================================

def build_attack_graph(probs, input_data, logs_df, threshold=0.5):
    """
    Builds the final attack graph directly from probabilities and log data.
    - Filters edges above the threshold
    - Retrieves log metadata
    - Builds 'nodes' and 'edges' structures with probabilities
    - Sorts edges temporally by event timestamp
    """
    print(f"\nBuilding attack graph with threshold > {threshold}...")

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

    # --- Sort edges temporally ---
    def _time_key(e):
        t = e.get("timestamp", None)
        if pd.isna(t):
            return pd.Timestamp.max
        if isinstance(t, pd.Timestamp):
            return t
        try:
            return pd.to_datetime(t)
        except Exception:
            return pd.Timestamp.max

    edges_list = sorted(edges_list, key=_time_key)

    # --- Build node metadata ---
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

    print(f"Attack graph built with {len(nodes_dict)} nodes and {len(edges_list)} edges (sorted chronologically).")
    return {"nodes": nodes_dict, "edges": edges_list}


# ===========================================================
# ---------------- SUMMARIZATION FOR LLM --------------------
# ===========================================================

def summarize_graph_for_prompt(graph, max_nodes=10, max_edges=40):
    """
    Condenses the graph into a more informative summary for the LLM.
    Now fully deterministic across runs for the same input graph.
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    if not edges or not nodes:
        return {"sample_nodes": [], "sample_edges": []}

    # Sort edges and nodes deterministically by ID before processing
    edges = sorted(edges, key=lambda e: (e.get("source_log_id", ""), e.get("dest_log_id", "")))
    node_ids_sorted = sorted(nodes.keys())

    # --- Build a deterministic directed graph ---
    G = nx.DiGraph()
    for e in edges:
        src, dst = e["source_log_id"], e["dest_log_id"]
        prob = float(e.get("edge_prob", 0.5))
        G.add_edge(src, dst, weight=-math.log(prob + 1e-6))

    # --- Compute deterministic centralities ---
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True, endpoints=True)

    # Combine into single score with deterministic tie-breaking by node ID
    scores = []
    for n in node_ids_sorted:
        deg = degree_centrality.get(n, 0)
        bet = betweenness.get(n, 0)
        scores.append((n, (deg + bet) / 2))

    # Sort deterministically: first by score desc, then by node ID asc
    ranked_nodes = sorted(scores, key=lambda x: (-x[1], x[0]))
    top_nodes = [n for n, _ in ranked_nodes[:max_nodes]]

    # --- Select top edges deterministically ---
    top_edges = [
        e for e in edges
        if e["source_log_id"] in top_nodes and e["dest_log_id"] in top_nodes
    ]
    top_edges = sorted(top_edges, key=lambda e: (-e.get("edge_prob", 0.0),
                                                e["source_log_id"], e["dest_log_id"]))[:max_edges]

    # --- Prepare nodes ---
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

    # --- Prepare edges ---
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


# ===========================================================
# ------------------- LLM REPORTING -------------------------
# ===========================================================

def generate_report_prompt(graph_summary):
    schema = """
    Your output MUST be ONLY valid JSON and follow this schema exactly:
    {
    "attack_name": "<short formal name of the detected attack type>",
    "attack_summary": "<technical summary of 4-5 sentences>",
    "nist_csf_mapping": {
        "Identify": "<identification stage details>",
        "Protect": "<defensive controls bypassed>",
        "Detect": "<what detections occurred>",
        "Respond": "<recommended immediate response>",
        "Recover": "<recommended recovery/hardening>"
    },
    "recommended_actions": [
        "<specific mitigation step 1>",
        "<specific mitigation step 2>",
        "<specific mitigation step 3>"
    ]
    }
    """
    graph_json = json.dumps(graph_summary, indent=2, default=str)
    return f"""
    You are a cybersecurity incident response analyst.
    Analyze the following summarized LOGS GRAPH, which represents relationships
    between security log events detected by an intrusion detection system.
    Provide a formal, concise technical report of the likely attack type and
    mapping to the NIST Cybersecurity Framework.
    Base your reasoning ONLY on the data provided and output strictly formatted JSON.
    --- BEGIN GRAPH SUMMARY ---
    {graph_json}
    --- END GRAPH SUMMARY ---
    {schema}
    """


def generate_attack_report_local(graph_summary, model_name="llama3.2"):
    """Generate a structured report via Ollama (local LLM)."""
    prompt = generate_report_prompt(graph_summary)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name, 
                "prompt": prompt,
                # "options": {
                #     "temperature": 0.0,   # deterministic decoding
                #     "num_predict": 512
                # }
            },
            timeout=180
        )

        text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                text += data.get("response", "")
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {"raw_output": text}
    except Exception as e:
        return {"error": str(e)}


def display_attack_report(report):
    print("\n--- AI-GENERATED ATTACK REPORT ---")
    if "error" in report:
        print("Error:", report["error"])
    elif "raw_output" in report:
        print(report["raw_output"])
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    print("------------------------------------")


# ===========================================================
# ------------------------- MAIN ----------------------------
# ===========================================================

def main():
    logs_df = load_dataset()
    node_features_np, edge_index_np, log_ids_np, in_channels = load_graph()
    model = load_model(in_channels=in_channels)
    input_data = prepare_data(node_features_np, edge_index_np, log_ids_np)

    print("\nRunning evaluation on the graph...")
    probs = eval_graph(model, input_data)
    print(f"Computed probabilities for {len(probs)} edges.")

    graph = build_attack_graph(probs, input_data, logs_df, threshold=0.9)

    if graph:
        print("\nSummarizing attack graph for LLM report...")
        graph_summary = summarize_graph_for_prompt(graph)
        print(f"Summary includes {graph_summary['important_nodes']} nodes and {graph_summary['important_edges']} edges.")

        print("\nGenerating AI Report using Local LLM...")
        report = generate_attack_report_local(graph_summary, model_name="llama3.2")
        display_attack_report(report)
    else:
        print("No attack edges above threshold — skipping report generation.")


if __name__ == "__main__":
    main()
