import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_scatter
from torch_geometric.nn.inits import glorot

from models import DGI, GraphCL
from layers import AvgReadout

class ConditionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        initialize Condition Net (CN)

        Args:
        - input_dim (int): input feature dim
        - hidden_dim (int): hidden dim of CN
        - output_dim (int): output dim
        - num_layers (int): CN layers
        - dropout (float): dropout ratio
        """
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

    def forward(self, x):
        x = F.elu(self.input_fc(x))
        for layer in self.hidden_fc:
            x = F.elu(layer(x))
        output = self.output_fc(x)
        return output

class downprompt(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nb_classes, think_layer_num, condition_layer_num):
        super(downprompt, self).__init__()
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, in_dim).cuda()
        self.embed_prompt = Embed_prompt(in_dim)

        # Condition Net
        self.condition_layers = nn.ModuleList([ConditionNet(in_dim, hid_dim, out_dim, condition_layer_num) for _ in range(think_layer_num)])
        self.condition_layers_num = condition_layer_num
        self.think_layer_num = think_layer_num

        # please refer to ablation experiment
        self.gcn_weight1 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.gcn_weight2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.gcn_weight3 = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, edge_index, gcn, idx, labels, train):        
        origin_x = x
        weight1 = self.gcn_weight1
        weight2 = self.gcn_weight2
        weight3 = self.gcn_weight3

        # each gcn layer has its own weight
        for condition_net in self.condition_layers:
            embed_1 = gcn.convs[0](x, edge_index)
            embed_2 = gcn.convs[1](embed_1, edge_index) + embed_1
            embed_3 = gcn.convs[2](embed_2, edge_index) + embed_2
            embed = weight1 * embed_1 + weight2 * embed_2 + weight3 * embed_3
            prompt = condition_net(embed)
            x = prompt * origin_x
        
        embed = gcn(x, edge_index)
        embed = self.embed_prompt(embed)

        rawret = embed[idx].cuda()  # rawret: idx_num x 256
        num = rawret.shape[0]
        if train == 1:
            self.ave = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
        ret = torch.FloatTensor(num, self.nb_classes).cuda()

        for x in range(0, num):
            for i in range(self.nb_classes):
                ret[x][i] = torch.cosine_similarity(rawret[x], self.ave[i], dim=0)
        ret = F.softmax(ret, dim=1)
        return ret
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class Embed_prompt(nn.Module):
    def __init__(self, in_channels: int):
        super(Embed_prompt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(5, in_channels))
        self.a = nn.Linear(in_channels, 5)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def forward(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p