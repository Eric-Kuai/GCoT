import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import AvgReadout 
import numpy as np
from sklearn.decomposition import PCA

def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
    return res

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout, sample = None):
        super(PrePrompt, self).__init__()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.negative_sample = torch.tensor(sample, dtype=torch.int64).cuda()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        g = self.gcn(x, edge_index)
        loss = compareloss(g, self.negative_sample, temperature=1)
        return loss

    def embed(self, x, edge_index):
        g = self.gcn(x, edge_index)
        return g.detach()

def mygather(feature, index): 
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

def compareloss(feature,tuples,temperature):
    h_tuples = mygather(feature,tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp = temp.cuda()
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def prompt_pretrain_sample(edge_index, n):
    nodenum = edge_index.max().item() + 1  
    adj_dict = {i: set() for i in range(nodenum)}
    for i, j in edge_index.t().tolist():
        adj_dict[i].add(j)
        adj_dict[j].add(i)

    res = np.zeros((nodenum, 1 + n), dtype=int)
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        neighbors = list(adj_dict[i])
        non_neighbors = np.setdiff1d(whole, neighbors)
        if len(neighbors) == 0:
            res[i][0] = i
        else:
            res[i][0] = neighbors[0]
        np.random.shuffle(non_neighbors)
        res[i][1:1 + n] = non_neighbors[:n]
        
    return res