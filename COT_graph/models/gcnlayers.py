import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp
from torch_geometric.nn import GCNConv,GraphConv
import tqdm
import numpy as np


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j
    

class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ELU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCNConv(hidden_dim, hidden_dim)
            else:
                nn = GCNConv(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, x , edge_index):

        for i in range(self.num_layers_num):
            # print("i",i)
            
            if i:
                # print(graph_output.shape)
                graph_output = self.convs[i](graph_output , edge_index) + graph_output
            else:
                graph_output = self.convs[i](x , edge_index)
            graph_output = self.act(graph_output)

        return graph_output
    
# def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
#     bsz = graph_sizes.size(0)
#     dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

#     min_size, max_size = graph_sizes.min(), graph_sizes.max()
#     mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

#     if min_size == max_size:
#         return batched_graph_feats.view(bsz, max_size, -1), mask
#     else:
#         graph_sizes_list = graph_sizes.view(-1).tolist()
#         unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
#         for i, l in enumerate(graph_sizes_list):
#             if l == max_size:
#                 continue
#             elif l > max_size:
#                 unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
#             else:
#                 mask[i, l:].fill_(0)
#                 zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
#                 unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
#         return torch.stack(unbatched_graph_feats, dim=0), mask
