#%%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import json
from Sampling import undersample

#%%

def oversample(data):
    labels = data.y.cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    minority_label = min(label_counts, key=label_counts.get)
    majority_count = max(label_counts.values())

    minority_indices = np.where(labels == minority_label)[0]
    additional_minority_indices = np.random.choice(minority_indices, majority_count - label_counts[minority_label])

    final_indices = np.concatenate((np.arange(data.num_nodes), additional_minority_indices))

    new_mask = torch.zeros(len(final_indices), dtype=torch.bool)
    new_mask[:data.num_nodes] = True

    return final_indices, new_mask
#%%
with open('DutchSchoolDataset/json/net1.json', 'r') as f:
    data = json.load(f)

#%%
x = torch.tensor(data['x'], dtype=torch.float)
edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
y = torch.tensor(data['y'][0], dtype=torch.long)

import torch
import numpy as np

def generate_random_mask(num_nodes, train_ratio=0.6, val_ratio=0.2):
    # 生成一个全节点的索引
    all_indices = np.arange(num_nodes)

    # 随机打乱索引
    np.random.shuffle(all_indices)

    # 根据比例划分训练、验证和测试集
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)
    num_test = num_nodes - num_train - num_val

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:num_train + num_val]
    test_indices = all_indices[num_train + num_val:]

    # 创建训练、验证和测试掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 将掩码置为True
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

# 假设图中有num_nodes个节点
num_nodes = 26

# 生成随机掩码
train_mask, val_mask, test_mask = generate_random_mask(num_nodes, train_ratio=0.6, val_ratio=0.2)
print(train_mask, val_mask, test_mask)

#%%
graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# train_mask = undersample(graph_data)
print(train_mask)
#%%
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GATConv(in_channels=x.size(1), out_channels=16)
        self.conv2 = GATConv(16, 5)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        # x = self.dropout(x)
        return F.log_softmax(x, dim=1)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%%
from torch.nn import functional as F
from collections import Counter
def get_weighted_loss(data, max_weight=10.0):
    labels = data.y.cpu().numpy()
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = {label: min(total_count / count, max_weight) for label, count in label_counts.items()}
    weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float).to(device)
    print(f'Class Weights: {weights}')
    return weights

# 使用加权损失
label_counts = Counter(graph_data.y.tolist())
class_weights = get_weighted_loss(graph_data, max_weight=1.)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], graph_data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

#%%
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# %%
import torch
from sklearn.model_selection import KFold
import numpy as np

def cross_validation(graph_data, k_folds=5, epochs=200):
    kf = KFold(n_splits=k_folds, shuffle=True)
    all_acc = []

    for fold, (train_index, test_index) in enumerate(kf.split(graph_data.x)):
        print(f'Fold {fold + 1}/{k_folds}')
        train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        train_mask[train_index] = True
        test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        test_mask[test_index] = True
        model = GCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data)
            loss = F.nll_loss(out[train_mask], graph_data.y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        out = model(graph_data)
        pred = out.argmax(dim=1)
        correct = (pred[test_mask] == graph_data.y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        all_acc.append(acc)
        print(f'Test Accuracy: {acc:.4f}')
    avg_acc = np.mean(all_acc)
    print(f'Average Accuracy over {k_folds} folds: {avg_acc:.4f}')
    return avg_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph_data = graph_data.to(device)

cross_validation(graph_data, k_folds=5)

# %%
# visualization
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.patches as mpatches

def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    labels = data.y.cpu().numpy()

    unique_labels = set(labels)
    cmap = plt.get_cmap('Set1')
    handles = [mpatches.Patch(color=cmap(i), label=f'Class {i}') for i in unique_labels]
    plt.legend(handles=handles, loc="best", title="Node Classes")
    nx.draw(G, pos, with_labels=True, node_color=data.y.cpu().numpy(), cmap=plt.get_cmap('Set1'), node_size=500, font_size=10)
    plt.show()

visualize_graph(graph_data)
# %%
import torch

def get_predictions_on_full_dataset(model, data):
    model.eval()

    data = data.to(device)

    with torch.no_grad(): 
        out = model(data)
    predictions = out.argmax(dim=1)
    
    return predictions


predictions = get_predictions_on_full_dataset(model, graph_data)
print("Predictions for the entire dataset:")
print(predictions.cpu().numpy())

# %%
print(y.cpu().numpy())
# %%
correct = (predictions == y).sum().item()
accuracy = correct / y.size(0)
print(f'Accuracy: {accuracy:.4f}')

#%%
