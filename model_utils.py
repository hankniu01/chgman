import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import GATConv
# from torch_geometric.data import Data, Batch

from comp import *


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''
    def __init__(self, in_dim = 300, mem_dim = 300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v) # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature) # (N, L, D)
        Q = self.linear(Q) # (N, L, D)
        feature = self.linear(feature) # (N, L, D)

        att_feature = torch.cat([feature, Q], dim = 2) # (N, L, 2D)
        att_weight = self.fc(att_feature) # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out



class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x



class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob = 0.1, bias = True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = F.gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        # x = torch.nn.functional.softmax(x,dim=1)
        return x

