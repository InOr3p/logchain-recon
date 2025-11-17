import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random

from training_edge_predictor import EdgePredictorGNN

# -----------------------------------------------
# --- CONFIG
# -----------------------------------------------

# Folders
TRAINVAL_GRAPH_DATA_DIR = "graphs_dataset_balanced"  # used for train+val in training
FULL_GRAPH_DATA_DIR = "graphs_dataset"               # full graphs folder (includes train/val+others)
MODELS_DIR = "models"                                # where .pth files are stored

# Model hyperparameters (must match training!)
HIDDEN_CHANNELS = 128
BATCH_SIZE = 32

MODELS_TO_LOAD = ["gnn_edge_predictor_gat_10epochs.pth", "gnn_edge_predictor_gat_20epochs.pth",
                  "gnn_edge_predictor_gcn_10epochs.pth", "gnn_edge_predictor_gcn_20epochs.pth",
                  "gnn_edge_predictor_sage_10epochs.pth", "gnn_edge_predictor_sage_20epochs.pth",]


# -----------------------------------------------
# --- DATASET
# -----------------------------------------------

class AttackGraphDataset(Dataset):
    """
    Loads the .npz graph files (same as in training script).
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.graph_files = sorted(glob.glob(os.path.join(root, "*.npz")))
        super(AttackGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.graph_files]

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        filepath = self.graph_files[idx]
        with np.load(filepath) as graph:
            x = torch.tensor(graph['node_feats'], dtype=torch.float32)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_label = torch.tensor(graph['edge_label'], dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, edge_label=edge_label, num_nodes=x.shape[0])
        return data


def build_test_dataset(test_ratio=0.2, seed=42):
    """
    Build a test dataset from FULL_GRAPH_DATA_DIR ('graphs_dataset') by excluding
    all graphs that are in TRAINVAL_GRAPH_DATA_DIR ('graphs_dataset_balanced'),
    then randomly selecting test_ratio of the remaining graphs for testing.
    """
    # Graphs used in training/validation (balanced subset)
    balanced_files = sorted(glob.glob(os.path.join(TRAINVAL_GRAPH_DATA_DIR, "*.npz")))
    balanced_basenames = {os.path.basename(f) for f in balanced_files}

    # All graphs from full dataset
    full_dataset = AttackGraphDataset(root=FULL_GRAPH_DATA_DIR)

    # Filter out any graph that was part of the balanced train/val subset
    candidate_test_files = [
        f for f in full_dataset.graph_files
        if os.path.basename(f) not in balanced_basenames
    ]

    if len(candidate_test_files) == 0:
        raise RuntimeError(
            "No candidate test graphs found: all graphs in 'graphs_dataset' "
            "seem to be part of 'graphs_dataset_balanced'."
        )

    # Randomly select 20% of the remaining graphs
    random.seed(seed)
    num_test = max(1, int(len(candidate_test_files) * test_ratio))
    test_files = random.sample(candidate_test_files, num_test)

    # Restrict the dataset to only the selected test files
    full_dataset.graph_files = sorted(test_files)

    print(f"Candidate graphs (excluding train/val): {len(candidate_test_files)}")
    print(f"Test set: {len(full_dataset)} graphs "
          f"({test_ratio * 100:.0f}% of remaining graphs)")

    return full_dataset

# -----------------------------------------------
# --- EVAL / TEST HELPERS
# -----------------------------------------------

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.edge_label)

            total_loss += loss.item() * batch.num_graphs
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch.edge_label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, accuracy, precision, recall, f1


def infer_gnn_type_from_filename(model_path: str) -> str:
    """
    Simple heuristic: infer gnn_type from filename.
    """
    name = os.path.basename(model_path).lower()
    if "sage" in name:
        return "sage"
    if "gcn" in name:
        return "gcn"
    if "gat" in name:
        return "gat"
    return "gat"

# -----------------------------------------------
# --- MAIN
# -----------------------------------------------

def main():
    print("Starting GNN Edge Predictor Testing...\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build test dataset and loader
    test_dataset = build_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get input dimensionality from dataset
    in_channels = test_dataset.num_node_features
    print(f"Detected {in_channels} input node features.\n")

    # Loss function (same as in training)
    criterion = nn.BCEWithLogitsLoss()

    results = []

    for model_name in MODELS_TO_LOAD:
        print("=" * 70)
        print(f"Testing model: {os.path.basename(model_name)}")
        gnn_type = infer_gnn_type_from_filename(model_name)
        print(f"Inferred gnn_type: {gnn_type}")
        model_path = os.path.join(MODELS_DIR, model_name)

        # Instantiate model
        model = EdgePredictorGNN(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            gnn_type=gnn_type,
        ).to(device)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Evaluate on test set
        test_loss, test_acc, test_prec, test_rec, test_f1 = eval_epoch(
            model, test_loader, criterion, device
        )

        print(f"  Test Loss     : {test_loss:.4f}")
        print(f"  Test Accuracy : {test_acc:.4f}")
        print(f"  Test Precision: {test_prec:.4f}")
        print(f"  Test Recall   : {test_rec:.4f}")
        print(f"  Test F1-score : {test_f1:.4f}")

        results.append({
            "model": os.path.basename(model_name),
            "gnn_type": gnn_type,
            "loss": test_loss,
            "acc": test_acc,
            "prec": test_prec,
            "rec": test_rec,
            "f1": test_f1,
        })

if __name__ == "__main__":
    main()
