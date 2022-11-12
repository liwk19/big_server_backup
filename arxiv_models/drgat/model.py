import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
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
        return output, gate


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1, feat_drop=0.0, attn_drop=0.0, edge_drop=0.0,
        negative_slope=0.2, use_attn_dst=True, residual=False, activation=None,
        allow_zero_in_degree=False, use_symmetric_norm=False, ngnn=False, num_expert=1, top_k=1, 
        expert_drop=0.0, gate='naive'
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:  # 我们是这种
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:       # 我们是这种
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0  # not implemented
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0  # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if ngnn:
            if num_expert > 1:
                self.ngnn_fc = FMoEMLP(num_heads * out_feats, num_heads * out_feats, num_expert=num_expert, top_k=top_k, 
                    expert_drop=expert_drop, gate=gate)
            else:
                self.ngnn_fc = nn.Linear(num_heads * out_feats, num_heads * out_feats)
        else:
            self.register_buffer("ngnn_fc", None)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if isinstance(self.ngnn_fc, nn.Linear):
            nn.init.xavier_normal_(self.ngnn_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            
            if graph.is_block:
                h_dst = h_src[: graph.number_of_dst_nodes()]
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:      # 我们是这种
                h_dst = h_src
                feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:     # 我们是这种情况
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm
            
            # ngnn
            if self.ngnn_fc is not None:
                rst = rst.view(rst.shape[0], self._num_heads * self._out_feats)
                rst, gate = self.ngnn_fc(rst)
                gate_std = torch.std(gate, 1).mean()
                rst = rst.view(rst.shape[0], self._num_heads, self._out_feats)
            else:
                gate_std = torch.tensor([0.0])

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst, gate_std


class DRGAT(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, n_heads, activation, dropout=0.0,
        hid_drop=0.0, input_drop=0.0, attn_drop=0.0, edge_drop=0.0,
        use_attn_dst=True, use_symmetric_norm=False, num_expert=16, top_k=1, moe_widget=[],
        gate='naive', expert_drop=0
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            if 'ngnn' in moe_widget and i > 0 and i < n_layers - 1:
                self.convs.append(
                    GATConv(in_hidden, out_hidden, num_heads=num_heads, attn_drop=attn_drop,
                        edge_drop=edge_drop, use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm, residual=True, ngnn=True, num_expert=num_expert, top_k=top_k, 
                        expert_drop=expert_drop, gate=gate
                    )
                )
            else:
                self.convs.append(
                    GATConv(in_hidden, out_hidden, num_heads=num_heads, attn_drop=attn_drop,
                        edge_drop=edge_drop, use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm, residual=True, ngnn=False
                    )
                )
            self.norms.append(nn.BatchNorm1d(n_heads * n_hidden))

        self.alpha_fc = nn.Linear(n_hidden * n_heads, n_hidden)
        self.rnn = nn.RNN(n_hidden, 1, 2)
        # self.lstm = nn.LSTM(n_hidden, 1, 2)

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.hid_drop = hid_drop
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation
        self.num_expert = num_expert

    def init_variables(self, number):
        alpha_hidden = torch.FloatTensor(2, number, 1).cuda()
        bound = 1 / math.sqrt(self.n_hidden)
        return nn.init.uniform_(alpha_hidden, -bound, bound)

    def forward(self, graph, feat):
        h_hidden = self.init_variables(feat.shape[0])
        h = self.input_drop(feat)

        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(), device=graph.device)
            self.perms.append(perm)

        _hidden = []
        h, _ = self.convs[0](graph, h, self.perms[0])
        h = h.flatten(1, -1)
        _hidden.append(h)
        gate_std_means = None

        for i in range(1, self.n_layers - 1):
            # graph.requires_grad = False
            h1 = F.normalize(_hidden[0], p=2, dim=1)
            hl = F.normalize(h, p=2, dim=1)
            h_mul = torch.mul(h1, hl).unsqueeze(0)
            h_mul_fc = self.alpha_fc(h_mul)
            alpha_out, h_hidden = self.rnn(h_mul_fc, h_hidden)
            alpha_out = torch.abs(alpha_out)
            alpha_evo = alpha_out.squeeze(0)

            h = F.relu(self.norms[i](h))
            h = F.dropout(h, p=self.hid_drop, training=self.training)
            h, gate_std_mean = self.convs[i](graph, h, self.perms[i])
            h = h.flatten(1, -1)
            gate_std_mean = gate_std_mean.unsqueeze(0)
            if gate_std_means == None:
                gate_std_means = gate_std_mean
            else:
                gate_std_means = torch.concat((gate_std_means, gate_std_mean))
            h = (1 - alpha_evo) * h + alpha_evo * _hidden[0]

        h = self.norms[-1](h)
        h = self.activation(h, inplace=True)
        h = self.dp_last(h)
        h, _ = self.convs[-1](graph, h, self.perms[-1])
        h = h.mean(1)
        h = self.bias_last(h)
        return h, gate_std_means.mean()
