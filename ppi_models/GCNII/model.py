import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fmoe_image import FMoE
from fmoe.linear import FMoELinear


class _Expert(nn.Module):
    def __init__(self, num_expert, d_input, d_output):
        super().__init__()
        self.fc = FMoELinear(num_expert, d_input, d_output, bias=True, rank=0)

    def forward(self, inp, fwd_expert_count):
        x = self.fc(inp, fwd_expert_count)
        return x


class FMoEMLP(FMoE):
    def __init__(self, d_input, d_output, num_expert=16, top_k=1, dropout=0, expert_drop=0, 
        gate='naive', expert_dp_comm="none"):
        super().__init__(num_expert=num_expert, d_model=d_input, top_k=top_k, world_size=1, 
            expert_drop=expert_drop, gate=gate)
        self.experts = _Expert(num_expert, d_input, d_output)
        self.dropout = nn.Dropout(dropout)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp, graph=None):
        output, gate = super().forward(inp, graph)
        output = self.dropout(output)
        return output


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False, num_expert=8, top_k=2,
        gate='naive', ngnn=False, expert_drop=0.0):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        if ngnn:
            if num_expert > 1:
                self.ngnn_fc = FMoEMLP(out_features, out_features, num_expert=num_expert, top_k=top_k, 
                    gate=gate, expert_drop=expert_drop)
            else:
                self.ngnn_fc = nn.Linear(out_features, out_features)
        else:
            self.ngnn_fc = None
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.ngnn_fc is not None:
            output = self.ngnn_fc(output)
        if self.residual:
            output = output + input
        return output


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, num_expert,
        top_k, gate, moe_widget, expert_drop):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            if 'ngnn' in moe_widget and i < nlayers - 1 and i > 0:   # 第1层和最后1层不加ngnn
                self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True,
                    num_expert=num_expert, top_k=top_k, gate=gate, ngnn=True, expert_drop=expert_drop))
            else:
                self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True, ngnn=False))
        
        if 'input_fc' in moe_widget:
            self.input_fc = FMoEMLP(nfeat, nhidden, num_expert=num_expert, top_k=top_k, gate=gate, expert_drop=expert_drop)
        else:
            self.input_fc = nn.Linear(nfeat, nhidden)
        if 'output_fc' in moe_widget:
            self.output_fc = FMoEMLP(nhidden, nclass, num_expert=num_expert, top_k=top_k, gate=gate, expert_drop=expert_drop)
        else:
            self.output_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.input_fc(x))
        h0 = layer_inner
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, h0, self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.output_fc(layer_inner))
        return layer_inner
