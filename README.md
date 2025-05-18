# Introduction
We provide the code (in pytorch) and datasets for our paper 
**"GCoT: Chain-of-Thought Prompt Learning for Graphs"** accepted by *SIGKDD* 2025.

## Description
The repository is organised as follows:
- **COT_node/**: implements pre-training and downstream tasks at the node level.
- **COT_graph/**: implements pre-training and downstream tasks at the garph level.

## Package Dependencies
* python 3.9
* torch 2.5.1
* cuda 12.4
* torch_geometric 2.6.1

## Generating data
You can find all data we use in *data.zip* in folders node and graph respectively.

In addition, we also provide 'data generate' file in which you are able to check and split data.
But attention, this file is an example for generating node data.
```python
dataset_name = 'xxx'
```
Just change the dataset you want and that will be fine.

## Running experiments
### Node Classification
Default dataset is Cora. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets. 

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.
```python
IF_PRETRAIN = 0
```
In the folder "COT_node", run:
- python execute.py

### Graph Classification
Default dataset is MUTAG. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets.

Pretrain and downstream test are in the same file, You can find a pretrain flag to control whether pretrain or not.
```python
IF_PRETRAIN = 0
```
In the folder "COT_graph", run:
- python execute.py
