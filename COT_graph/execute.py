
import argparse
parser = argparse.ArgumentParser("COT")
parser.add_argument('--dataset', type=str, default="MUTAG", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--down_type', type=str, default='GPFplus', help='GPFplus or Graphprompt or none')
parser.add_argument('--useCOT', type=str, default='yes', help='yes or no')
parser.add_argument('--expand_dim', type=str, default='yes', help='yes or no')
parser.add_argument('--premodel', type=str, default='LP', help='the type of pretrain model')   
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 

import numpy as np
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt
import preprompt

from downprompt import downprompt
import csv
from tqdm import tqdm



print('-' * 100)
print(args)
print('-' * 100)

aug_type = args.aug_type
drop_percent = args.drop_percent

seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))

from torch_geometric.datasets import QM9
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

nb_epochs = 10000
patience = 50
lr = 0.00001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
useMLP = False
class_num = 2
LP = False
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
firstbest = 0

threeD_dataset=['qm9']
TU_Dataset = ['ENZYMES','PROTEINS','BZR','COX2','MUTAG']
def load_dataset(name):
    if name in TU_Dataset:
        dataset = TUDataset(root='data', name=name,use_node_attr=True)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return dataset

cnt_wait = 0
num_layers_num = 3
pretrain_datasets = [args.dataset]
print(pretrain_datasets)
num_pretrain_dataset_num = len(pretrain_datasets)


datasetload = load_dataset(dataset)
pretrain_loaders = DataLoader(datasetload, batch_size = len(datasetload), shuffle=False)
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Build pretrain dataset string
pretrain_dataset_str = ''
for str in pretrain_datasets:
    pretrain_dataset_str += '_' + str
# Define checkpoint save path
save_dir = os.path.join(current_dir, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.join(save_dir, f'model_node_{args.premodel}_{pretrain_dataset_str}.pkl')
# Instantiate model and optimizer                                                                          
# Flag for pretraining
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

IF_PRETRAIN = False
if IF_PRETRAIN:
    best = 1e9
    cnt_wait = 0
    patience = 10  # Define the patience for early stopping
    
    # Pretraining loop\
    for step, data in enumerate(pretrain_loaders):
        data = data.cuda()
        
        # Extract necessary data for training
        batch = data.batch
        num_node_attributes = range(datasetload.num_node_attributes)
        print(num_node_attributes)
        x = data.x
        # x = data.x[:,num_node_attributes]
        feature_dim = x.shape[1]
        num_nodes = x.size(0)
        
        # Generate sparse edge index based on node positions (using radius graph)
        edge_index = data.edge_index.cuda()

        negetive_sample = preprompt.prompt_pretrain_sample(edge_index.cpu(), 150)

        model = PrePrompt(feature_dim,hid_units, num_layers_num=num_layers_num, dropout=0.1,sample=negetive_sample,premodel=args.premodel)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        
        for epoch in tqdm(range(nb_epochs)):         
            total_loss = 0
            model.train()
            optimizer.zero_grad()
            loss = model(x, edge_index)

            # print("loss:",loss)
            loss.backward()
            optimizer.step()
        if loss < best:
            best = loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        
        # Early stopping condition
        if cnt_wait == patience:
            print('Early stopping!')
            break
        
        # Optionally log the training progress
        print(f"Epoch {epoch + 1}/{nb_epochs}, Loss: {loss:.4f}")

print('#'*50)
print('PreTrain datasets are ', pretrain_datasets)
print('Downastream dataset is ', args.dataset)



condition_dim_list = [8]
condition_num_list = [2]
think_num_list = [2]
a_list = [0.1]
b_list = [0.001]
c_list = [1]

condition_dim = 4
condition_num = 1
think_num = 2
print(f'loading model from {save_name}')

for step, data in enumerate(pretrain_loaders):
    print(data)
    label = data.y
    labels=np.array(data.y)
    print(labels)
    np.unique(labels)                                                               

    nb_classes=len(np.unique(labels))
    data = data.cuda()

    batch = data.batch
    print(batch)
    print(np.unique(batch.shape))
    num_node_attributes = range(datasetload.num_node_attributes)
    x = data.x
    print("x",x)
    feature_dim = x.shape[1]
    num_nodes = x.size(0)
    edge_index = data.edge_index


    model = PrePrompt(feature_dim,hid_units, num_layers_num=num_layers_num, dropout=0.1,sample=1)


    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
    model = model.cuda()
    embeds = model.embed(x, edge_index)
    downstreamlrlist = [0.001]
    acclist = torch.FloatTensor(100,).cuda()


    for condition_dim in condition_dim_list:
        for condition_num in condition_num_list:
            for think_num in think_num_list:
                for downstreamlr in downstreamlrlist:
                    
                    tot = torch.zeros(1)
                    tot = tot.cuda()
                    accs = []
                    print('-' * 100)

                    for shotnum in range(1,2):
                        idx_test = torch.load("data/fewshot_{}_graph/testset/idx.pt".format(args.dataset.lower())).squeeze().type(torch.long).cuda()
                        print(idx_test)
                        test_lbls = torch.load("data/fewshot_{}_graph/testset/labels.pt".format(args.dataset.lower())).squeeze().type(torch.long).squeeze().cuda()
                        print(test_lbls)
                        # print(test_lbls)
                        test_lbls = label[idx_test].cuda()
                        print(label[idx_test])

                        tot = torch.zeros(1)
                        tot = tot.cuda()
                        accs = []
                        macrof = []
                        microf = []


                        print("shotnum",shotnum)
                        for i in tqdm(range(100)):
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            log = downprompt(hid_units,condition_dim,feature_dim,nb_classes,think_layer_num=think_num,condition_layer_num=condition_num,type=args.down_type,usecot=args.useCOT,expand=args.expand_dim)
                            for name, param in log.named_parameters():
                                # print(f"Name: {name}")
                                param.requires_grad = True
                            log.train()

                            idx_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).squeeze().type(torch.long).cuda()

                            cnt_wait = 0
                            best = 1e9
                            best_t = 0

                            print(idx_train)
                            train_lbls = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).squeeze().type(torch.long).squeeze().cuda()
                            print(train_lbls)
                            print(labels[idx_train.cpu()])
                            
                            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr,weight_decay=0.0001)
                            log = log.cuda()
                            best = 1e9
                            pat_steps = 0
                            best_acc = torch.zeros(1)
                            best_acc = best_acc.cuda()
                            for _ in range(400):
                                opt.zero_grad()

                                logits = log(x,edge_index,model.gcn,idx_train,batch,train_lbls,1).float()
                                # print(logits)
                                loss = xent(logits, train_lbls)
                                if loss < best:
                                    best = loss
                                    # best_t = epoch
                                    cnt_wait = 0
                                    # torch.save(model.state_dict(), args.save_name)
                                else:
                                    cnt_wait += 1
                                if cnt_wait == patience:
                                    # print('Early stopping!')
                                    break
                                
                                loss.backward()
                                opt.step()

                            logits = log(x,edge_index,model.gcn,idx_test,batch)
                            # print(log.a)
                            preds = torch.argmax(logits, dim=1).cuda()
                            print('preds:',preds)
                            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                            preds_cpu = preds.cpu().numpy()
                            test_lbls_cpu = test_lbls.cpu().numpy()
                            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
                            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')
                            microf.append(micro_f1 * 100)
                            macrof.append(macro_f1 * 100)
                            accs.append(acc * 100)
                            print('acc:[{:.4f}]'.format(acc))
                            tot += acc
                        print('-' * 100)
                        print('Average accuracy:[{:.4f}]'.format(tot.item() / 100))
                        accs = torch.stack(accs)
                        acc_mean = accs.mean().item()
                        acc_std = accs.std().item()
                        microf_mean = sum(microf) / len(microf)
                        macrof_mean = sum(macrof) / len(macrof)
                        microf_std = torch.std(torch.tensor(microf)).item()
                        macrof_std = torch.std(torch.tensor(macrof)).item() 
                        print('Mean:[{:.2f}]'.format(acc_mean))
                        print('Std :[{:.2f}]'.format(acc_std))
                        print('macrof_mean:[{:.2f}]'.format(macrof_mean))
                        print('macrof_std :[{:.2f}]'.format(macrof_std))
                        print('-' * 100)
                        row = [args.useCOT,args.expand_dim,args.down_type,args.dataset,condition_dim,condition_num,think_num,shotnum,lr,downstreamlr,hid_units,acc_mean,acc_std,microf_mean,microf_std,macrof_mean,macrof_std]
                        out = open("data/data_{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(row)
