import numpy as np
import os
import random
from torch_geometric.datasets import TUDataset
import torch

# Specify the dataset name
dataset_name = 'ENZYMES'  # Change this to any dataset you wish to use

# Load dataset
dataset = TUDataset(root='data', name=dataset_name, use_node_attr=True)

labels = np.array([data.y.item() for data in dataset])
unique_labels = np.unique(labels)
nb_classes = len(unique_labels)
print('Number of classes:', nb_classes)
print('Labels:', unique_labels)

shotnumlist = [1]  # List of k values for k-shot learning

total_num = len(labels)
test_size = 100  # Fix the test set to be the last 1000 graphs
if test_size > total_num:
    test_size = total_num  # Ensure test_size does not exceed total number of graphs

# Fixed test set indices (last 1000 graphs)
test_indices = list(range(total_num - test_size, total_num))
test_labels = [labels[idx] for idx in test_indices]

# Convert test_indices and test_labels to tensors
test_indices = torch.tensor(test_indices)
test_labels = torch.tensor(test_labels)

# Remaining indices for training (excluding test_indices)
remaining_indices = list(range(0, total_num - test_size))
remaining_labels = labels[:total_num - test_size]

for shotnum in shotnumlist:
    for i in range(100):  # Generate 100 datasets
        # For each class, sample k instances from remaining_indices
        train_indices = []
        train_labels = []
        for label in unique_labels:
            # Get indices of data points with this label in remaining_indices
            indices = [idx for idx in remaining_indices if labels[idx] == label]
            # Randomly select k instances
            if len(indices) >= shotnum:
                selected_indices = random.sample(indices, shotnum)
            else:
                # If fewer instances than k, use all available
                selected_indices = indices
            train_indices.extend(selected_indices)
            train_labels.extend([label]*len(selected_indices))
        # Convert train_indices and train_labels to tensors
        train_indices = torch.tensor(train_indices)
        train_labels = torch.tensor(train_labels)
        # Save the train_indices and train_labels
        save_dir = "data/fewshot_{}_graph/{}-shot_{}/{}/".format(
            dataset_name.lower(), shotnum, dataset_name.lower(), i)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_indices, os.path.join(save_dir, 'index.pt'))
        torch.save(train_labels, os.path.join(save_dir, 'labels.pt'))

    # Save the fixed test set
    test_dir = "data/fewshot_{}_graph/{}-shot_{}/testset/".format(
        dataset_name.lower(), shotnum, dataset_name.lower())
    os.makedirs(test_dir, exist_ok=True)
    torch.save(test_indices, os.path.join(test_dir, 'index.pt'))
    torch.save(test_labels, os.path.join(test_dir, 'labels.pt'))

print("end")
