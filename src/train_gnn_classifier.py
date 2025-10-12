import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
class Config:
    METADATA_DIR = "metadata"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 20
    NUM_WORKERS = 4 # Speeds up data loading
    HIDDEN_CHANNELS = 128

print(f"Using device: {Config.DEVICE}")

# --- 1. EFFICIENT DATASET LOADER ---
# This class loads graphs one by one from disk, which is memory-efficient.
class AlertGraphDataset(Dataset):
    def __init__(self, metadata_path):
        super().__init__()
        self.metadata = pd.read_csv(metadata_path)

    def len(self):
        return len(self.metadata)

    def get(self, idx):
        # Get graph info from the metadata file
        graph_info = self.metadata.iloc[idx]
        filepath = graph_info['filepath']
        label = graph_info['label_id']

        # Load the graph data from the .npz file
        with np.load(filepath) as data:
            node_feats = torch.from_numpy(data['node_feats']).float()
            edge_index = torch.from_numpy(
                np.vstack([data['sources'], data['destinations']])
            ).long()

        # Create a PyG Data object
        graph = Data(x=node_feats, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        return graph

# --- 2. GNN MODEL DEFINITION ---
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GraphSAGE is a robust and scalable GNN layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        # Classifier part
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN message passing
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        # Pool all node features into a single graph-level feature vector
        x = global_mean_pool(x, batch)

        # Classification
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x

# --- 3. TRAINING & EVALUATION LOOPS ---
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(Config.DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(Config.DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    accuracy = correct / len(loader.dataset)
    return accuracy, all_preds, all_labels

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    # Load label mapping
    with open(os.path.join(Config.METADATA_DIR, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Load datasets
    train_dataset = AlertGraphDataset(os.path.join(Config.METADATA_DIR, 'train.csv'))
    val_dataset = AlertGraphDataset(os.path.join(Config.METADATA_DIR, 'val.csv'))
    test_dataset = AlertGraphDataset(os.path.join(Config.METADATA_DIR, 'test.csv'))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Initialize model
    # Get input feature dimension from the first graph
    num_node_features = train_dataset[0].num_node_features
    model = GNNClassifier(
        in_channels=num_node_features,
        hidden_channels=Config.HIDDEN_CHANNELS,
        out_channels=num_classes
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    for epoch in range(1, Config.EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        val_acc, _, _ = test(model, val_loader)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print("\n--- Training Finished ---")

    # Final evaluation on the test set
    test_acc, test_preds, test_labels = test(model, test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Display detailed classification report
    class_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(test_labels, test_preds, target_names=class_names)
    print("\n--- Classification Report ---")
    print(report)

    # Save the trained model
    model_path = "gnn_attack_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTrained model saved to {model_path}")