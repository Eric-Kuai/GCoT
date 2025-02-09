from torch_geometric.datasets import WikipediaNetwork

dataset_chameleon = WikipediaNetwork(root="data", name="chameleon")[0]


# 加载Squirrel数据集
dataset_squirrel = WikipediaNetwork(root="data", name="squirrel")[0]
num_classes_chameleon = dataset_chameleon.y.unique().size(0)
num_classes_squirrel = dataset_squirrel.y.unique().size(0)
# 打印Chameleon数据集信息
print("Chameleon:")
print(f"nodes: {dataset_chameleon.num_nodes}")
print(f"edges: {dataset_chameleon.num_edges}")
print(f"features: {dataset_chameleon.num_node_features}")
print(f"classes: {num_classes_chameleon}")

# 打印Squirrel数据集信息
print("\nSquirrel:")
print(f"nodes: {dataset_squirrel.num_nodes}")
print(f"edges: {dataset_squirrel.num_edges}")
print(f"features: {dataset_squirrel.num_node_features}")
print(f"classes: {num_classes_squirrel}")