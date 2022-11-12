import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fmoe_image import FMoE
from fmoe.linear import FMoELinear
import collections


class _Expert(nn.Module):
    def __init__(self, num_expert, d_input, d_output):
        super().__init__()
        self.fc = FMoELinear(num_expert, d_input, d_output, bias=True, rank=0)

    def forward(self, inp, fwd_expert_count):
        x = self.fc(inp, fwd_expert_count)
        return x


class FMoEMLP(FMoE):
    def __init__(self, d_input, d_output, num_expert=16, top_k=1, dropout=0, expert_drop=0, 
        gate='naive', expert_dp_comm="none", experts=None):
        super().__init__(num_expert=num_expert, d_model=d_input, top_k=top_k, world_size=1, 
            expert_drop=expert_drop, gate=gate)
        if experts == None:
            self.experts = _Expert(num_expert, d_input, d_output)
        else:
            self.experts = experts
        self.dropout = nn.Dropout(dropout)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp, graph=None):
        inp = inp.float()
        output, gate = super().forward(inp, graph)
        output = self.dropout(output)
        return output


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, att_drop=0., act='none', moe=False, num_expert=1, top_k=1, expert_drop=0.0, gate='naive'):
        super(Transformer, self).__init__()
        print('n_channels:', n_channels)
        self.query ,self.key, self.value = [self._conv(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        # if moe:
        #     self.query ,self.key, self.value = [FMoEMLP(n_channels, c, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate) for c in (n_channels//8, n_channels//8, n_channels)]
        # else:
        #     self.query ,self.key, self.value = [nn.Linear(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def _conv(self,n_in, n_out):
        return torch.nn.utils.spectral_norm(nn.Conv1d(n_in, n_out, 1, bias=False))

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        # x_n = x.transpose(1, 2)
        # ori_shape0 = x_n.shape[0]
        # x_n = x_n.view(-1, x_n.shape[2]).float()
        f, g, h = self.query(x), self.key(x), self.value(x)
        # f, g, h = f.view(ori_shape0, -1, f.shape[1]).transpose(1, 2), g.view(ori_shape0, -1, g.shape[1]).transpose(1, 2), h.view(ori_shape0, -1, h.shape[1]).transpose(1, 2)
        beta = F.softmax(self.act(torch.bmm(f.transpose(1,2), g)), dim=1)
        beta = self.att_drop(beta)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, moe=False, num_expert=1, top_k=1, 
                    expert_drop=0.0, gate='naive'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.moe = moe
        self.groups = groups
        if moe:
            self.fc = FMoEMLP(groups * cin, cout, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)
        else:
            if not bias:
                self.bias = None
            self.W = nn.Parameter(torch.randn(groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(groups, self.cout))

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        if not self.moe:
            gain = nn.init.calculate_gain("relu")
            xavier_uniform_(self.W, gain=gain)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.moe:
            x = x.view(x.shape[0], -1)
            x_list = [x.clone()] * self.groups
            x = torch.concat(x_list)
            x = self.fc(x)
            x = x.reshape(-1, self.groups, x.shape[1])
            return x
        else:
            return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias


class L2Norm(nn.Module):
    def __init__(self, dim):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SeHGNN_mag(nn.Module):
    def __init__(self, dataset, nfeat, hidden, nclass, num_feats, num_label_feats, tgt_key,
                 dropout, input_drop, att_drop, label_drop, n_layers_1, n_layers_2, n_layers_3,
                 act, residual=False, bns=False, label_bns=False, label_residual=True, moe_widget=[],
                 num_expert=1, top_k=1, expert_drop=0.3, gate='naive'):
        super(SeHGNN_mag, self).__init__()
        self.dataset = dataset
        self.residual = residual
        self.tgt_key = tgt_key
        self.label_residual = label_residual
        self.moe_widget = moe_widget
        self.num_label_feats = num_label_feats
        self.num_expert = num_expert
        
        self.feat_project_layers1 = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats, bias=True, moe=('conv1d' in moe_widget), num_expert=num_expert, top_k=top_k, 
                expert_drop=expert_drop, gate=gate),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        if 'feat_ngnn' in moe_widget:
            if num_expert == 1:
                self.feat_ngnn = nn.Linear(hidden, hidden)   # myalter
            else:
                self.feat_ngnn = FMoEMLP(hidden, hidden, num_expert=num_expert, top_k=top_k, 
                    expert_drop=expert_drop, gate=gate)
        self.feat_project_layers2 = nn.Sequential(
            Conv1d1x1(hidden, hidden, num_feats, bias=True, moe=('conv1d' in moe_widget), num_expert=num_expert, top_k=top_k, 
                expert_drop=expert_drop, gate=gate),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        if num_label_feats > 0:
            self.label_feat_project_layers1 = nn.Sequential(
                Conv1d1x1(nclass, hidden, num_label_feats, bias=True, moe=('conv1d' in moe_widget), num_expert=num_expert, top_k=top_k, 
                    expert_drop=expert_drop, gate=gate),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
            if 'label_ngnn' in moe_widget:
                if num_expert == 1:
                    self.label_ngnn = nn.Linear(hidden, hidden)   # myalter
                else:
                    self.label_ngnn = FMoEMLP(hidden, hidden, num_expert=num_expert, top_k=top_k, 
                        expert_drop=expert_drop, gate=gate)
            self.label_feat_project_layers2 = nn.Sequential(
                Conv1d1x1(hidden, hidden, num_label_feats, bias=True, moe=('conv1d' in moe_widget), num_expert=num_expert, top_k=top_k, 
                    expert_drop=expert_drop, gate=gate),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.label_feat_project_layers1 = None

        self.semantic_aggr_layers = Transformer(hidden, att_drop, act, moe=('transformer' in moe_widget), 
            num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)
        if 'ngnn1' in moe_widget:
            if num_expert == 1:
                self.ngnn1 = nn.Linear((num_feats + num_label_feats) * hidden, (num_feats + num_label_feats) * hidden)
            else:
                self.ngnn1 = FMoEMLP((num_feats + num_label_feats) * hidden, (num_feats + num_label_feats) * hidden,
                    num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)
        if 'concat' in moe_widget:
            self.concat_project_layer = FMoEMLP((num_feats + num_label_feats) * hidden, hidden, 
                num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)
        else:
            self.concat_project_layer = nn.Linear((num_feats + num_label_feats) * hidden, hidden)

        if 'ngnn2' in moe_widget:
            if num_expert == 1:
                self.ngnn2 = nn.Linear(hidden, hidden)   # myalter
            else:
                self.ngnn2 = FMoEMLP(hidden, hidden,
                    num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)

        if self.residual:
            if 'residual' in moe_widget:
                self.res_fc = FMoEMLP(nfeat, hidden, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)
            else:
                self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        def add_nonlinear_layers(nfeats, dropout, bns=False):
            if bns:
                return [
                    nn.BatchNorm1d(hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]
            else:
                return [
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ]

        if 'lr_output' in moe_widget:
            lr_output_layers = [
                [FMoEMLP(hidden, hidden, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)] + add_nonlinear_layers(hidden, dropout, bns)
                for _ in range(n_layers_2-1)]
            self.lr_output = nn.Sequential(*(
                [ele for li in lr_output_layers for ele in li] + [
                FMoEMLP(hidden, nclass, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate),
                nn.BatchNorm1d(nclass)]))
        else:
            lr_output_layers = [
                [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
                for _ in range(n_layers_2-1)]
            self.lr_output = nn.Sequential(*(
                [ele for li in lr_output_layers for ele in li] + [
                nn.Linear(hidden, nclass, bias=False),
                nn.BatchNorm1d(nclass)]))

        if self.label_residual:   # 我们是这种
            if 'label_residual' in moe_widget:
                label_fc_layers = [
                    [FMoEMLP(hidden, hidden, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)] + add_nonlinear_layers(hidden, dropout, bns)
                    for _ in range(n_layers_3-2)]
                self.label_fc = nn.Sequential(*(
                    [FMoEMLP(nclass, hidden, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)] + add_nonlinear_layers(hidden, dropout, bns) \
                    + [ele for li in label_fc_layers for ele in li] + [FMoEMLP(hidden, nclass, num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate)]))
            else:
                label_fc_layers = [
                    [nn.Linear(hidden, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns)
                    for _ in range(n_layers_3-2)]
                self.label_fc = nn.Sequential(*(
                    [nn.Linear(nclass, hidden, bias=not bns)] + add_nonlinear_layers(hidden, dropout, bns) \
                    + [ele for li in label_fc_layers for ele in li] + [nn.Linear(hidden, nclass, bias=True)]))
            self.label_drop = nn.Dropout(label_drop)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        if 'feat_ngnn' in self.moe_widget and self.num_expert == 1:
            nn.init.xavier_uniform_(self.feat_ngnn.weight, gain=gain)
        for layer in self.feat_project_layers1:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        for layer in self.feat_project_layers2:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        if self.label_feat_project_layers1 is not None:
            for layer in self.label_feat_project_layers1:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()
            for layer in self.label_feat_project_layers2:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()
        
        if self.num_label_feats > 0:
            if 'label_ngnn' in self.moe_widget and self.num_expert == 1:
                nn.init.xavier_uniform_(self.label_ngnn.weight, gain=gain)

        if 'concat' not in self.moe_widget:
            nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
            nn.init.zeros_(self.concat_project_layer.bias)
        if 'ngnn1' in self.moe_widget and self.num_expert == 1:
            nn.init.xavier_uniform_(self.ngnn1.weight, gain=gain)       # myalter
        if 'ngnn2' in self.moe_widget and self.num_expert == 1:
            nn.init.xavier_uniform_(self.ngnn2.weight, gain=gain)       # myalter

        if self.residual and 'residual' not in self.moe_widget:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        if 'lr_output' not in self.moe_widget:
            for layer in self.lr_output:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        if 'label_residual' not in self.moe_widget:
            if self.label_residual:
                for layer in self.label_fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=gain)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def forward(self, feats_dict, layer_feats_dict, label_emb):
        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)
        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))
        x = self.feat_project_layers1(x)
        if 'feat_ngnn' in self.moe_widget:
            ori_shape0 = x.shape[0]
            x = x.view(-1, x.shape[2])
            x = self.feat_ngnn(x)
            x = x.view(ori_shape0, -1, x.shape[1])
        x = self.feat_project_layers2(x)

        if self.num_label_feats > 0:
            label_feats = self.input_drop(torch.stack(list(layer_feats_dict.values()), dim=1))
            label_feats = self.label_feat_project_layers1(label_feats)
            if 'label_ngnn' in self.moe_widget:
                ori_shape0 = label_feats.shape[0]
                label_feats = label_feats.view(-1, label_feats.shape[2])
                label_feats = self.label_ngnn(label_feats)
                label_feats = label_feats.view(ori_shape0, -1, label_feats.shape[1])
            label_feats = self.label_feat_project_layers2(label_feats)
            x = torch.cat((x, label_feats), dim=1)

        x = self.semantic_aggr_layers(x.transpose(1,2)).reshape(B, -1)
        if 'ngnn1' in self.moe_widget:
            x = self.ngnn1(x)
        x = self.concat_project_layer(x)
        if 'ngnn2' in self.moe_widget:
            x = self.ngnn2(x)

        if self.residual:
            x = x + self.res_fc(tgt_feat)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)
        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))
        return x
