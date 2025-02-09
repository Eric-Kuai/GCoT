import numpy as np
import os
import random
from torch_geometric.datasets import Planetoid, Amazon, WebKB, WikipediaNetwork
import torch

# choose your dataset
dataset_name = 'Cora'
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo']
WebKB_datasets = ['Wisconsin']
WikipediaNetwork_datasets = ['Squirrel']
if dataset_name in Planetoid_datasets:
    dataset = Planetoid(root='data', name=dataset_name)
elif dataset_name in Amazon_datasets:
    dataset = Amazon(root='data', name=dataset_name)
elif dataset_name in WebKB_datasets:
    dataset = WebKB(root='data', name=dataset_name)
elif dataset_name in WikipediaNetwork_datasets:
    dataset = WikipediaNetwork(root='data', name=dataset_name)
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")

data = dataset[0]
labels = np.array(data.y)
unique_labels = np.unique(labels)
nb_classes = len(unique_labels)
print('Number of classes:', nb_classes)
print('Labels:', unique_labels)

# test pool
node_num = data.num_nodes
test_size = 1000 # you have to change the value according to particular datasets
test_indices = list(range(node_num - test_size, node_num))
test_labels = [labels[idx] for idx in test_indices]
test_indices = torch.tensor(test_indices)
test_labels = torch.tensor(test_labels)

# training pool
remaining_indices = list(range(0, node_num - test_size))
remaining_labels = labels[:node_num - test_size]

# generate k-shot files
shotnum_list = [1]
for shotnum in shotnum_list:
    for i in range(100):
        train_indices = []
        train_labels = []
        for label in unique_labels:
            indices = [idx for idx in remaining_indices if labels[idx] == label]
            if len(indices) >= shotnum:
                selected_indices = random.sample(indices, shotnum)
            else:
                selected_indices = indices
            train_indices.extend(selected_indices)
            train_labels.extend([label] * len(selected_indices))
        train_indices = torch.tensor(train_indices)
        train_labels = torch.tensor(train_labels)

        # Save the train_indices and train_labels
        save_dir = "data/fewshot_{}_node/{}-shot/{}/".format(dataset_name.lower(), shotnum, i)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_indices, os.path.join(save_dir, 'index.pt'))
        torch.save(train_labels, os.path.join(save_dir, 'labels.pt'))

    # Save the fixed test set
    test_dir = "data/fewshot_{}_node/{}-shot/testset/".format(dataset_name.lower(), shotnum)
    os.makedirs(test_dir, exist_ok=True)
    torch.save(test_indices, os.path.join(test_dir, 'index.pt'))
    torch.save(test_labels, os.path.join(test_dir, 'labels.pt'))

print("end")