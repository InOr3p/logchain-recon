import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

EXTRACT_DIR = 'extracted_dataset'

# --- Model Config ---
HIDDEN_CHANNELS = 128

class EdgePredictorGNN(nn.Module):
    """
    GNN that classifies edges, structured as an Encoder-Decoder.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout_p=0.3):
        super(EdgePredictorGNN, self).__init__()
        self.dropout_p = dropout_p
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.encoder_convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if i < len(self.encoder_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def decode(self, z, edge_index):
        node_src_embeds = z[edge_index[0]]
        node_dst_embeds = z[edge_index[1]]
        edge_embeds = torch.cat([node_src_embeds, node_dst_embeds], dim=1)
        return self.decoder(edge_embeds)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_index)
        return logits.squeeze(-1)

def get_log_details_from_data(log_id, data):
    return data['id'] 

def edge_logits_to_probs(logits):
    """
    This was the missing function.
    For binary edge classification, this is typically the sigmoid function.
    """
    return torch.sigmoid(logits)


def eval_graph(model, data, device='cpu'):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)  # shape [E]
        probs = edge_logits_to_probs(logits)
        return probs

def reconstruct_attack_sequence(probs, input_data, log_dataset, threshold=0.5):
    """
    Filters edges, looks up log details, and builds the sorted sequence.
    """
    print(f"\nReconstructing sequence using threshold > {threshold}...")
    
    predictions = (probs > threshold).long()
    
    # Find the *indices* (positions) of the attack edges
    attack_indices = (predictions == 1).nonzero(as_tuple=True)[0]

    num_attacks = attack_indices.numel()
    if num_attacks == 0:
        print("No attack edges detected at this threshold.")
        return []

    print(f"Found {num_attacks} predicted attack events.")
    
    # Filter the edge_index to get only the (src, dst) pairs for attacks
    attack_edge_index = input_data.edge_index[:, attack_indices]
    
    try:
        log_ids_array = input_data.log_ids
    except AttributeError:
        print("Error: 'input_data.log_ids' not found.")
        print("Please ensure 'log_ids' was loaded from the .npz file and attached to the Data object.")
        return []

    log_sequence = []
    print("Retrieving log details for attack edges...")
    for i in range(num_attacks):
        src_idx = attack_edge_index[0, i].item()
        dst_idx = attack_edge_index[1, i].item()
        
        src_log_id = log_ids_array[src_idx]
        dst_log_id = log_ids_array[dst_idx]
        
        try:
            src_log_details = log_dataset.loc[src_log_id]
            
            event_time = src_log_details['@timestamp']

        except KeyError as e:
            print(f"Warning: Log ID {e} (or its pair) not found in log dataset.")
            raise KeyError

        log_sequence.append({
            "timestamp": event_time,
            "source_log_id": src_log_id,
            "dest_log_id": dst_log_id,
        })

    print("Sorting sequence by timestamp...")
    sorted_sequence = sorted(log_sequence, key=lambda k: k['timestamp'])
    
    return sorted_sequence


def load_dataset():
    print("\nLoading dataset...")
    logs_path = os.path.join(EXTRACT_DIR, "sorted_ds_with_labels.parquet")

    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Missing file: {logs_path}")

    logs_df = pd.read_parquet(logs_path)

    # ----------------------------
    # CLEAN AND PREPARE DATA
    # ----------------------------

    logs_df['id'] = logs_df['id'].astype(str)
    logs_df = logs_df.set_index('id')

    # ensure timestamps are datetime
    logs_df['@timestamp'] = pd.to_datetime(logs_df['@timestamp'], errors='coerce')

    return logs_df


def load_model(in_channels):
    # 3a. Instantiate the model architecture
    print("Instantiating model structure...")
    model = EdgePredictorGNN(
        in_channels=in_channels, 
        hidden_channels=HIDDEN_CHANNELS
    )

    # 3b. Load the saved weights (the "state dictionary")
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

    # The shape is [num_nodes, num_features]
    in_channels = node_features_np.shape[1]
    
    print(f"\nSuccessfully detected {in_channels} input features (in_channels).")

    return node_features_np, edge_index_np, log_ids_np, in_channels


def prepare_data(node_features_np, edge_index_np, log_ids_np):
    x_tensor = torch.tensor(node_features_np, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    input_data = Data(x=x_tensor, edge_index=edge_index_tensor)
    input_data.log_ids = log_ids_np

    print(f"\nSuccessfully created PyG Data object:")
    print(input_data)

    return input_data

def main():
    # --- LOAD DATASET ---
    logs_df = load_dataset()
    
    # --- LOAD GRAPH DATA ---
    node_features_np, edge_index_np, log_ids_np, in_channels = load_graph() 

    # --- LOAD THE MODEL ---
    model = load_model(in_channels=in_channels)

    # --- PREPARE DATA AND RUN INFERENCE ---
    input_data = prepare_data(node_features_np, edge_index_np, log_ids_np)

    print(input_data.log_ids)

    print("\nRunning evaluation on the graph...")
    probs = eval_graph(model, input_data)
    print(f"Probabilities calculated for {len(probs)} edges.")
    
    attack_sequence = reconstruct_attack_sequence(probs, input_data, logs_df, threshold=0.9)
    
    if attack_sequence:
        print("\n--- [Reconstructed Attack Sequence (Sorted)] ---")
        for i in range(10):
            step = attack_sequence[i]
            print(f"Time: {step['timestamp']} | SRC: {step['source_log_id']} -> DST: {step['dest_log_id']}")
            print("-" * 20)


if __name__ == "__main__":
    main()