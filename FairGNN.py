#%%

import torch
import networkx as nx
import numpy as np
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

#%%
# Load the Karate Club dataset
dataset = KarateClub()
data = dataset[0]

# Manually create train/val/test masks
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Assign masks
train_mask[:int(num_nodes * 0.6)] = True
val_mask[int(num_nodes * 0.6):int(num_nodes * 0.8)] = True
test_mask[int(num_nodes * 0.8):] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Assume we divide nodes into two groups based on their degree
degree = data.edge_index[0].bincount()  # Calculate node degrees
threshold = degree.median()  # Use median degree as threshold to create two groups
group_0 = (degree <= threshold)  # Low-degree group
group_1 = (degree > threshold)   # High-degree group

#%%
# Visualize the Karate Club graph and mark groups and prediction masks
G = nx.karate_club_graph()
plt.figure(figsize=(10, 8))

# Set node colors based on group membership and prediction masks
node_colors = []
for i in range(num_nodes):
    if data.train_mask[i]:
        if group_0[i]:
            node_colors.append('lightblue')
        else:
            node_colors.append('green')  # Training nodes
    elif data.val_mask[i]:
        if group_0[i]:
            node_colors.append('purple')
        else:
            node_colors.append('yellow')  # Validation nodes
    elif data.test_mask[i]:
        if group_0[i]:
            node_colors.append('blue')  # Test nodes in Group 0
        else:
            node_colors.append('red')   # Test nodes in Group 1
    else:
        node_colors.append('gray')  # Unlabeled nodes (if any)

# Draw the graph with font colors based on masks
pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, edge_color='gray', 
        labels={i: f'{i}' for i in range(num_nodes)}, font_color='black')

# Change font colors for test nodes to white
for i in range(num_nodes):
    if data.test_mask[i]:
        nx.draw_networkx_labels(G, pos, labels={i: f'{i}'}, font_color='white', font_size=10)

plt.title("Karate Club Graph with Training (Green), Validation (Yellow), and Test Nodes (Group 0: Blue, Group 1: Red)")
plt.show()

#%%
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
model = GCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

#%%
# Training the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

#%%
# Testing the model and evaluating fairness
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    # Accuracy for entire test set
    acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

    # Fairness evaluation: accuracy for each group
    group_0_acc = accuracy_score(data.y[group_0 & data.test_mask].cpu(), pred[group_0 & data.test_mask].cpu())
    group_1_acc = accuracy_score(data.y[group_1 & data.test_mask].cpu(), pred[group_1 & data.test_mask].cpu())

    return acc, group_0_acc, group_1_acc, pred

#%%
# Training loop
for epoch in range(200):
    loss = train()
    acc, group_0_acc, group_1_acc, pred = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Group 0 Acc: {group_0_acc:.4f}, Group 1 Acc: {group_1_acc:.4f}')


#%%
# Visualize the prediction results
plt.figure(figsize=(10, 8))

# Set node colors based on prediction results
pred_colors = []
for i in range(num_nodes):
    if data.train_mask[i]:
        pred_colors.append('green')  # Training nodes
    elif data.val_mask[i]:
        pred_colors.append('yellow')  # Validation nodes
    elif data.test_mask[i]:
        if pred[i] == data.y[i]:
            pred_colors.append('cyan')  # Correct prediction
        else:
            pred_colors.append('magenta')  # Incorrect prediction
    else:
        pred_colors.append('gray')  # Unlabeled nodes (if any)

# Draw the graph with prediction results
nx.draw(G, pos, with_labels=True, node_color=pred_colors, node_size=500, font_size=10, edge_color='gray', 
        labels={i: f'{i}' for i in range(num_nodes)}, font_color='black')

plt.title("Karate Club Graph with Prediction Results (Correct: Cyan, Incorrect: Magenta)")
plt.show()
#%%
# Visualize the ground truth labels
plt.figure(figsize=(10, 8))

# Set node colors based on ground truth labels
gt_colors = []
for i in range(num_nodes):
    if data.y[i] == 0:
        gt_colors.append('blue')  # Class 0
    else:
        gt_colors.append('red')  # Class 1

# Draw the graph with ground truth labels
nx.draw(G, pos, with_labels=True, node_color=gt_colors, node_size=500, font_size=10, edge_color='gray', 
        labels={i: f'{i}' for i in range(num_nodes)}, font_color='black')

plt.title("Karate Club Graph with Ground Truth Labels (Class 0: Blue, Class 1: Red)")
plt.show()
# %%



