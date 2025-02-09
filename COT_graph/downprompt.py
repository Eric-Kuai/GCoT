import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
from models import DGI, GraphCL
from layers import GCN, AvgReadout
import torch_scatter
from torch_geometric.nn.inits import glorot
class ConditionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(ConditionNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        # print("input_dim",input_dim)
        # print("hidden_dim",hidden_dim)
        # print("output_dim",output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        for layer in self.hidden_fc:
            x = layer(x)
        output = self.output_fc(x)
        return output

class downprompt(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nb_classes, think_layer_num, condition_layer_num,type,usecot,expand):
        super(downprompt, self).__init__()
        if expand == 'yes':
            self.condition_layers = nn.ModuleList([ConditionNet(in_dim*2, hid_dim, out_dim, condition_layer_num) for _ in range(think_layer_num)])
        elif expand == 'no':
            self.condition_layers = nn.ModuleList([ConditionNet(in_dim, hid_dim, out_dim, condition_layer_num) for _ in range(think_layer_num)])
        self.nb_classes = nb_classes
        self.condition_layers_num = condition_layer_num
        self.think_layer_num = think_layer_num
        self.layer_norm = nn.LayerNorm(out_dim)
        self.condition_net = ConditionNet(in_dim, hid_dim, in_dim, condition_layer_num)
        self.preprompt = downstreamprompt(out_dim)
        self.graphprompt = downstreamprompt(in_dim)
        self.GPFplus = GPFplusAtt(in_dim,p_num=5)
        self.a = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.type = type
        self.usecot = usecot
        self.expand = expand

    def forward(self, x, edge_index, gcn, idx, batch, labels=None, train=0):


        
        origin_x = x
        if self.usecot == 'yes':
            for condition_net in self.condition_layers:
                node_embed0 = gcn.convs[0](x , edge_index)
                graph_embed0 = torch_scatter.scatter(src=node_embed0, index=batch, dim=0, reduce='mean')
                graph_embed0 = torch.index_select(graph_embed0, 0, batch)
                total_embed0 = torch.cat([node_embed0, graph_embed0], dim=-1)


                node_embed1 = gcn.convs[1](node_embed0 , edge_index) + node_embed0
                graph_embed1 = torch_scatter.scatter(src=node_embed1, index=batch, dim=0, reduce='mean')
                graph_embed1 = torch.index_select(graph_embed1, 0, batch)
                total_embed1 = torch.cat([node_embed1, graph_embed1], dim=-1)


                node_embed2 = gcn.convs[2](node_embed1 , edge_index) + node_embed1
                graph_embed2 = torch_scatter.scatter(src=node_embed2, index=batch, dim=0, reduce='mean')
                graph_embed2 = torch.index_select(graph_embed2, 0, batch)
                total_embed2 = torch.cat([node_embed2, graph_embed2], dim=-1)

                a = torch.sigmoid(self.a)
                b = torch.sigmoid(self.b)    
                if self.expand == 'yes':
                    graph_output = a *total_embed0 +b * total_embed1 + total_embed2
                elif self.expand == 'no':
                    graph_output = a *node_embed0 +b * node_embed1 +node_embed2
                
                prompt = condition_net(graph_output)
                x = prompt * origin_x
        embed = gcn(x, edge_index)
        if self.type == 'GPFplus':
            # print('useGPFplus')
            embed = self.GPFplus.add(embed)
        elif self.type == 'Graphprompt':
            # print('useGraphprompt')
            embed = self.graphprompt(embed)
        

        rawret = torch_scatter.scatter(src=embed, index=batch, dim=0, reduce='mean')
        rawret = rawret[idx]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)

        num = rawret.shape[0]
        ret = torch.zeros(num, self.nb_classes, device=rawret.device)  # Ensure device match
        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = F.normalize(rawret, dim=-1)  # Normalize for cosine similarity stability
        similarity = torch.mm(rawret, rawret.t())  # Compute pairwise similarity
        ret = similarity[:num, num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, graph_embedding):
        graph_embedding = self.weight * graph_embedding
        return graph_embedding

def averageemb(labels, rawret, nb_class):
    return torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p