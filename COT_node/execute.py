import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from preprompt import PrePrompt
import preprompt
from downprompt import downprompt

import argparse
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Amazon, WebKB, WikipediaNetwork
from torch_geometric.loader import DataLoader

# --------------------
# model parameters
# --------------------

parser = argparse.ArgumentParser("COT")
pretrain_dataset = 'Cora'
parser.add_argument('--dataset', type=str, default="Cora", help='data') # downstream dataset
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3.pkl', help='save ckpt name') 
args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

device = torch.device("cuda")
dataset = args.dataset
seed = args.seed
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# --------------------
# load datasets
# --------------------

# there are 8 available datasets in total
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo']
WebKB_datasets = ['Wisconsin']
WikipediaNetwork_datasets = ['Squirrel']
def load_dataset(name):
    if name in Planetoid_datasets:
        dataset = Planetoid(root='data', name=name)
    elif name in Amazon_datasets:
        dataset = Amazon(root='data', name=name)
    elif name in WebKB_datasets:
        dataset = WebKB(root='data', name=name)
    elif name in WikipediaNetwork_datasets:
        dataset = WikipediaNetwork(root='data', name=name)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return dataset

print('PreTrain datasets: ', pretrain_dataset)
datasetload = load_dataset(dataset)
batch = len(datasetload)
pretrain_loaders = DataLoader(datasetload, batch_size = batch, shuffle=False)

# --------------------
# file path
# --------------------

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
save_dir = os.path.join(current_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.join(save_dir, f'model_node_{pretrain_dataset}.pkl')

# --------------------
# pretrain
# --------------------

pretrain_layers_num = 3
pretrain_epoch = 100
pretrain_lr = 0.0001
hid_units = 256
patience = 20
best = 1e9
sparse = True
useMLP = False
LP = False
xent = nn.CrossEntropyLoss()

IF_PRETRAIN = 0 # pretrain or not
if IF_PRETRAIN:
    for step, data in enumerate(pretrain_loaders):
        data = data.cuda()
        x = data.x                      # node feature
        num_nodes = x.shape[0]          # node num
        feature_dim = x.shape[1]        # feature dim
        edge_index = data.edge_index    # edge
        negetive_sample = preprompt.prompt_pretrain_sample(edge_index, 100)
        model = PrePrompt(feature_dim, hid_units, pretrain_layers_num, dropout=drop_percent, sample=negetive_sample)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
        cnt_wait = 0
        
        for epoch in tqdm(range(pretrain_epoch)):
            model.train()
            optimizer.zero_grad()
            loss = model(x, edge_index)

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), save_name)
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{pretrain_epoch}, Loss: {loss:.4f}")

# --------------------
# downstream tasks
# --------------------

print('#'*50)
print('Downastream dataset: ', args.dataset)
print(f'loading model from {save_name}')

shotnum = 1               # k-shot
downstreamlr = 0.001
condition_hid_dim = 32    # CN hidden dim
condition_layer_num = 1   # CN layers
think_layer_num = 1       # think layers - 1
task_num = 100
down_epoch = 200
patience = 20
best = 1e9
print('-' * 100)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
tot = torch.zeros(1)
tot = tot.cuda()
accs = []
macrof = []
for step, data in enumerate(pretrain_loaders):
    labels = np.array(data.y)                                                             
    nb_classes = len(np.unique(labels))     # labels
    data = data.cuda()
    x = data.x
    num_node = x.shape[0]
    feature_dim = x.shape[1]
    edge_index = data.edge_index
    model = PrePrompt(feature_dim, hid_units, pretrain_layers_num, dropout=0.1, sample=1)
    model.load_state_dict(torch.load(save_name))
    x = x.cuda()
    edge_index = edge_index.cuda()
    model = model.cuda()

    # load testset
    idx_test = torch.load("data/fewshot_{}_node/{}-shot/testset/index.pt".format(args.dataset.lower(), shotnum)).squeeze().type(torch.long).cuda()
    test_lbls = torch.load("data/fewshot_{}_node/{}-shot/testset/labels.pt".format(args.dataset.lower(), shotnum)).squeeze().type(torch.long).squeeze().cuda()
    log = downprompt(hid_units, condition_hid_dim, feature_dim, nb_classes, think_layer_num, condition_layer_num)
    print("shotnum", shotnum)

    for i in tqdm(range(task_num)):
        idx_train = torch.load("data/fewshot_{}_node/{}-shot/{}/index.pt".format(args.dataset.lower(), shotnum, i)).squeeze().type(torch.long).cuda()
        train_lbls = torch.load("data/fewshot_{}_node/{}-shot/{}/labels.pt".format(args.dataset.lower(), shotnum, i)).squeeze().type(torch.long).squeeze().cuda()
        # idx_train: 1 x (k x class_num)
        # train_labels: 1 x (k x class_num)
        print("task number:", i)
        print("train node index:", idx_train)
        print("train labels:", train_lbls)
        opt = torch.optim.Adam([{'params': log.parameters()}], lr=downstreamlr)
        log = log.cuda()
        cnt_wait = 0

        for _ in range(down_epoch):
            log.train()
            opt.zero_grad()
            logits = log(x, edge_index, model.gcn, idx_train, train_lbls, train=1).float().cuda()
            loss = xent(logits, train_lbls)
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break
            loss.backward(retain_graph=True)
            opt.step()

        logits = log(x, edge_index, model.gcn, idx_test, labels=None, train=0)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        preds_cpu = preds.cpu().numpy()
        test_lbls_cpu = test_lbls.cpu().numpy()
        macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')
        macrof.append(macro_f1 * 100)
        accs.append(acc * 100)
        tot += acc

    # --------------------
    # record results
    # --------------------

    print('-' * 100)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / task_num))
    accs = torch.stack(accs)
    acc_mean = accs.mean().item()
    acc_std = accs.std().item()
    macrof_mean = sum(macrof) / len(macrof)
    macrof_std = torch.std(torch.tensor(macrof)).item() 
    print('Mean:[{:.2f}]'.format(acc_mean))
    print('Std :[{:.2f}]'.format(acc_std))
    print('macrof_mean:[{:.2f}]'.format(macrof_mean))
    print('macrof_std :[{:.2f}]'.format(macrof_std))
    print('-' * 100)
    row = [args.dataset, shotnum, pretrain_lr, pretrain_layers_num, hid_units, 
        downstreamlr, condition_hid_dim, condition_layer_num, think_layer_num,
        acc_mean, acc_std, macrof_mean, macrof_std]
    out = open("data/{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(row)