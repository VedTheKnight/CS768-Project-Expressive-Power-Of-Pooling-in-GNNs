import time
import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F

from scripts.nn_model import GIN_Pool_Net
from scripts.utils import EXPWL1Dataset, DataToFloat
from torch_geometric.loader import DataLoader

# === Useful Functions ===

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, aux_loss = model(data)
        loss = F.nll_loss(out, data.y) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        loss = F.nll_loss(out, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

# === Fixed base hyperparameters ===
epochs = 250
batch_size = 32
hidden_channels = 64
lr = 1e-4
runs = 1

# === Experiment settings ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
poolings = ['specpool', 'gmt', 'softpool', 'learnable-cluster']
default_pool_ratio = 0.1
default_num_layers_pre = 2
default_num_layers_post = 1

pool_ratios = [0.1, 0.25, 0.5]
num_layers_pre_list = [2, 3]
num_layers_post_list = [1, 2]

# === Load dataset ===
path = "data/EXPWL1/"
dataset = EXPWL1Dataset(path, transform=DataToFloat())

avg_nodes = int(dataset.data.num_nodes / len(dataset))
max_nodes = max(d.num_nodes for d in dataset)

rng = np.random.default_rng(1)
rnd_idx = rng.permutation(len(dataset))
dataset = dataset[list(rnd_idx)]

train_dataset = dataset[len(dataset) // 5:]
val_dataset = dataset[:len(dataset) // 10]
test_dataset = dataset[len(dataset) // 10:len(dataset) // 5]

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)

# === Run experiments ===
results = []
for pooling in poolings:
    # --- Vary pool_ratio (keep pre=2, post=1) ---
    for pool_ratio in pool_ratios:
        num_layers_pre = default_num_layers_pre
        num_layers_post = default_num_layers_post
        print(f"\n>>> {pooling} | pool_ratio={pool_ratio} | pre={num_layers_pre} | post={num_layers_post}")
        
        start = time.time()
        model = GIN_Pool_Net(
            in_channels=train_dataset.num_features,
            out_channels=train_dataset.num_classes,
            num_layers_pre=num_layers_pre,
            num_layers_post=num_layers_post,
            hidden_channels=hidden_channels,
            average_nodes=avg_nodes,
            pooling=pooling,
            pool_ratio=pool_ratio,
            max_nodes=max_nodes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_test_acc = 0

        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer)
            _, val_loss = test(model, val_loader)
            test_acc, _ = test(model, test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

        elapsed = time.time() - start
        results.append({
            "pooling": pooling,
            "pool_ratio": pool_ratio,
            "num_layers_pre": num_layers_pre,
            "num_layers_post": num_layers_post,
            "test_acc": best_test_acc,
            "time_sec": elapsed
        })

    # --- Vary num_layers_pre (keep pool_ratio=0.1, post=1) ---
    for num_layers_pre in num_layers_pre_list:
        pool_ratio = default_pool_ratio
        num_layers_post = default_num_layers_post
        print(f"\n>>> {pooling} | pool_ratio={pool_ratio} | pre={num_layers_pre} | post={num_layers_post}")
        
        start = time.time()
        model = GIN_Pool_Net(
            in_channels=train_dataset.num_features,
            out_channels=train_dataset.num_classes,
            num_layers_pre=num_layers_pre,
            num_layers_post=num_layers_post,
            hidden_channels=hidden_channels,
            average_nodes=avg_nodes,
            pooling=pooling,
            pool_ratio=pool_ratio,
            max_nodes=max_nodes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_test_acc = 0

        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer)
            _, val_loss = test(model, val_loader)
            test_acc, _ = test(model, test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

        elapsed = time.time() - start
        results.append({
            "pooling": pooling,
            "pool_ratio": pool_ratio,
            "num_layers_pre": num_layers_pre,
            "num_layers_post": num_layers_post,
            "test_acc": best_test_acc,
            "time_sec": elapsed
        })

    # --- Vary num_layers_post (keep pool_ratio=0.1, pre=2) ---
    for num_layers_post in num_layers_post_list:
        pool_ratio = default_pool_ratio
        num_layers_pre = default_num_layers_pre
        print(f"\n>>> {pooling} | pool_ratio={pool_ratio} | pre={num_layers_pre} | post={num_layers_post}")
        
        start = time.time()
        model = GIN_Pool_Net(
            in_channels=train_dataset.num_features,
            out_channels=train_dataset.num_classes,
            num_layers_pre=num_layers_pre,
            num_layers_post=num_layers_post,
            hidden_channels=hidden_channels,
            average_nodes=avg_nodes,
            pooling=pooling,
            pool_ratio=pool_ratio,
            max_nodes=max_nodes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_test_acc = 0

        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer)
            _, val_loss = test(model, val_loader)
            test_acc, _ = test(model, test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

        elapsed = time.time() - start
        results.append({
            "pooling": pooling,
            "pool_ratio": pool_ratio,
            "num_layers_pre": num_layers_pre,
            "num_layers_post": num_layers_post,
            "test_acc": best_test_acc,
            "time_sec": elapsed
        })

# === Save Results ===
df = pd.DataFrame(results)
df.to_csv("experiment_results_partial_sweep.csv", index=False)
print("\n✅ Partial hyperparameter sweep complete. Results saved to experiment_results_partial_sweep.csv.")
