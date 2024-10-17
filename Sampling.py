import numpy as np
import torch

def undersample(data, target_class_ratio=1.0):
    labels = data.y.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    majority_label = max(label_counts, key=label_counts.get)

    minority_count = min(label_counts.values())
    majority_count = int(minority_count * target_class_ratio)

    majority_indices = np.where(labels == majority_label)[0]
    minority_indices = np.where(labels != majority_label)[0]

    np.random.shuffle(majority_indices)
    sampled_majority_indices = majority_indices[:majority_count]

    final_indices = np.concatenate((sampled_majority_indices, minority_indices))

    new_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_mask[final_indices] = True

    return new_mask

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

