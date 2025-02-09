from torch_geometric.datasets import Planetoid, Amazon, WebKB, WikipediaNetwork

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

# basic info
print(f"Dataset name: {dataset.name}")
print(f"Graph number: {len(dataset)}")

# each of 8 datasets mentioned above only has one graph
data = dataset[0]
print(data)

# node info
print(f"Node num: {data.num_nodes}")
print(f"Feature dim: {dataset.num_node_features}")
print(f"Edge num: {data.num_edges}")
print(f"Self loop: {data.has_self_loops()}")
print(f"Undirected: {data.is_undirected()}")