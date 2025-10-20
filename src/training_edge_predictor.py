import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------
# --- HYPERPARAMETERS AND CONFIGURATION
# -----------------------------------------------

# --- Paths ---
GRAPH_DATA_DIR = "graphs_dataset_balanced"
MODEL_SAVE_PATH = os.path.join("models", "gnn_edge_predictor.pth")
PLOT_SAVE_PATH = os.path.join("models", "gnn_training_history.png")

# --- Model Config ---
HIDDEN_CHANNELS = 128
GNN_LAYERS = 2

# --- Training Config ---
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_SPLIT_RATIO = 0.8

# -----------------------------------------------
# --- PYTORCH GEOMETRIC DATASET
# -----------------------------------------------

class AttackGraphDataset(Dataset):
    """
    Loads the .npz graph files from your GraphBuilder.
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

# -----------------------------------------------
# --- GNN MODEL (Encoder-Decoder)
# -----------------------------------------------

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

# -----------------------------------------------
# --- TRAINING AND VALIDATION FUNCTIONS
# -----------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.edge_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(batch.edge_label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, accuracy, f1

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
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

# -----------------------------------------------
# --- PLOTTING FUNCTION
# -----------------------------------------------

def plot_training_history(history, save_path):
    """
    Generates and saves a plot of training and validation metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, 1.05) # Accuracy is between 0 and 1

    # Plot 3: F1 Score
    axes[2].plot(epochs, history['train_f1'], 'bo-', label='Training F1 Score')
    axes[2].plot(epochs, history['val_f1'], 'ro-', label='Validation F1 Score')
    axes[2].set_title('F1 Score (Positive Class)')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0, 1.05) # F1 is between 0 and 1

    fig.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(save_path)
    print(f"\nTraining plot saved to {save_path}")
    plt.show()

# -----------------------------------------------
# --- MAIN SCRIPT
# -----------------------------------------------
def main():
    print("Starting GNN Edge Predictor Training...\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and split dataset
    try:
        dataset = AttackGraphDataset(root=GRAPH_DATA_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}"); return
    if len(dataset) == 0:
        print("Error: No graphs found in dataset directory."); return
    
    print(f"Loaded {len(dataset)} graphs.")
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * TRAIN_SPLIT_RATIO)
    train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    print(f"Training set: {len(train_dataset)} graphs | Validation set: {len(val_dataset)} graphs")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Optimizer, Loss
    in_channels = dataset.num_node_features
    print(f"Detected {in_channels} input node features.")
    
    model = EdgePredictorGNN(in_channels, HIDDEN_CHANNELS, GNN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nModel Initialized. Starting training for {EPOCHS} epochs...\n")

    # --- Store history ---
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []
    }
    
    best_val_f1 = -1.0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_f1'].append(val_f1)

        print("-" * 50)
        print(f"Epoch {epoch:02d} / {EPOCHS:02d}")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Valid -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} (Pos Class)")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved (F1: {best_val_f1:.4f})")

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best model (F1: {best_val_f1:.4f}) saved to {MODEL_SAVE_PATH}")
    print("=" * 50)

    plot_training_history(history, PLOT_SAVE_PATH)


if __name__ == "__main__":
    main()