import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init
from fmoe_image import FMoE, FMoELinear


class _Expert2(nn.Module):
    def __init__(self, num_expert, d_input, d_hidden, d_output, activation, rank=0, bias=True):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_input, d_hidden, bias=bias, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_output, bias=bias, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class _Expert(nn.Module):
    def __init__(self, num_expert, d_input, d_output, rank=0, bias=True):
        super().__init__()
        self.htoh = FMoELinear(num_expert, d_input, d_output, bias=bias, rank=rank)

    def forward(self, inp, fwd_expert_count):
        x = self.htoh(inp, fwd_expert_count)
        return x


class FMoEMLP(FMoE):
    def __init__(self, d_input, d_output, num_expert=128, top_k=2, fmoe2=0, dropout=0, 
        gate='naive', bias=True, expert_drop=0, activation=torch.nn.GELU(),
        expert_dp_comm="none", expert_rank=0):
        super().__init__(num_expert=num_expert, d_model=d_input, top_k=top_k, world_size=1, gate=gate,
            expert_drop=expert_drop)
        if fmoe2 > 0:
            self.experts = _Expert2(num_expert, d_input, fmoe2, d_output, activation, rank=expert_rank, bias=bias)
        else:
            self.experts = _Expert(num_expert, d_input, d_output, rank=expert_rank, bias=bias)
        self.mark_parallel_comm(expert_dp_comm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, graph=None):
        output, gate = super().forward(inp, graph)    # positionwise feed-forward
        output = self.dropout(output)
        return output, gate


# GATv2卷积层
class GAT2Conv(nn.Module):
    def __init__(self, node_feats, edge_feats, out_feats, n_heads=1, attn_drop=0.0, edge_drop=0.0,
        negative_slope=0.2, residual=True, activation=None, use_attn_dst=True,
        allow_zero_in_degree=True, use_symmetric_norm=False, num_expert=4, top_k=2,
        n_hidden_layers=2, fmoe2=0, moe_drp=0, gate='naive', moe_widget=['ngnn'], fmoe22=0, expert_drop=0
    ):
        super(GAT2Conv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        # expand_as_pair把node_feats深拷贝成2份并返回
        self._out_feats = out_feats   # 为GAT的n_hidden
        self._allow_zero_in_degree = allow_zero_in_degree   # GAT's default: False
        self._use_symmetric_norm = use_symmetric_norm   # default: False
        self.moe_widget = moe_widget

        # feat fc
        if 'src_fc' in moe_widget:
            self.src_fc = FMoEMLP(self._in_src_feats, out_feats * n_heads, num_expert, 
                top_k, fmoe22 * n_heads, moe_drp, 'naive', bias=False, expert_drop=expert_drop)
        else:
            self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)  # 对进入的src点做一层转化
        if residual:
            if 'dst_fc' in moe_widget:
                self.dst_fc = FMoEMLP(self._in_src_feats, out_feats * n_heads, num_expert, 
                    top_k, fmoe22 * n_heads, moe_drp, 'naive', expert_drop=expert_drop)
            else:
                self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)   # 对进入的dst点做转化
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_fc = nn.Linear(out_feats * n_heads, n_heads, bias=False)  # attn_fc之后获得的为n_heads维度的

        if edge_feats > 0:
            if 'attn_edge_fc' in moe_widget:
                self.attn_edge_fc = FMoEMLP(edge_feats, n_heads, num_expert, 
                    top_k, 0, moe_drp, 'naive', bias=False, expert_drop=expert_drop)         # 不支持fmoe2
            else:
                self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)   # edge也做
        else:
            self.attn_edge_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        # self.fc_adjs = nn.Linear(out_feats * n_heads, out_feats * n_heads)
        # self.fc_adjs2 = nn.Linear(out_feats * n_heads, out_feats * n_heads)   # 之后弄几个MLP
        self.ngnn_layers = nn.ModuleList()
        if 'ngnn' in moe_widget:
            self.ngnn_layers.append(FMoEMLP(out_feats * n_heads, out_feats * n_heads, num_expert, 
                top_k, fmoe2 * n_heads, moe_drp, gate, expert_drop=expert_drop))
        else:
            if fmoe2:
                self.ngnn_layers.append(nn.Linear(out_feats * n_heads, fmoe2 * n_heads))
                self.ngnn_layers.append(nn.Linear(fmoe2 * n_heads, out_feats * n_heads))
            else:
                self.ngnn_layers.append(nn.Linear(out_feats * n_heads, out_feats * n_heads))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):   # reset all parameters
        gain = nn.init.calculate_gain("relu")
        if 'src_fc' not in self.moe_widget:
            nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            if 'dst_fc' not in self.moe_widget:
                nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        for layer in self.ngnn_layers:
            if not isinstance(layer, FMoEMLP):
                nn.init.xavier_normal_(layer.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):     # 无用
        self._allow_zero_in_degree = set_value

    # graph为一个子图subgraphs[i]，feat_src为GAT里中途的点向量h，feat_edg为中途的边向量edge_emb
    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():  # 在local_scope里对graph的修改在离开local_scope后不会生效，这个很利于forward计算
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False
            if graph.is_block:  # 我们是这种，# 我们是这种。每个边有个src node，有个dst node
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
                # number_of_src_nodes()为(n+1)-hop邻居，number_of_dst_nodes()为n-hop邻居
            else:   
                feat_dst = feat_src

            if self._use_symmetric_norm:    # default: false
                degs = graph.srcdata["deg"]
                # degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm   # 对邻接矩阵做权重正则化

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)  # 留着residual connection用

            graph.srcdata.update({"feat_src_fc": feat_src_fc})
            graph.dstdata.update({"feat_dst_fc": feat_dst_fc})
            graph.srcdata.update({"feat_src_fc_gate": feat_src_fc.view(-1, self._n_heads * self._out_feats)})
            graph.dstdata.update({"feat_dst_fc_gate": feat_dst_fc.view(-1, self._n_heads * self._out_feats)})
            
            attn_src = self.leaky_relu(feat_src_fc.view(-1, self._n_heads * self._out_feats))
            attn_src = self.attn_fc(attn_src).view(-1, self._n_heads, 1)
            attn_dst = self.leaky_relu(feat_dst_fc.view(-1, self._n_heads * self._out_feats))
            attn_dst = self.attn_fc(attn_dst).view(-1, self._n_heads, 1)
            graph.srcdata.update({"attn_src": attn_src})
            graph.dstdata.update({"attn_dst": attn_dst})
            graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))  # 两个相加得到attn_node
            
            e = graph.edata["attn_node"]
            # e = self.leaky_relu(e)    # 这是标准GATv2做法，但会爆显卡！
            # e = self.attn_fc(e).view(-1, self._n_heads, 1)
            if feat_edge is not None:   # 我们是这种
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)  # 输出维度为n_heads
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]  # e就是边本身的attn和边连接的两点的attention之和

            if self.training and self.edge_drop > 0:   # drop edge
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing: fn.u_mul_e为message generation，把点的feat_src_fc属性乘以边的a属性，赋给点的m属性
            # fn.sum为message reduction/aggregation，把某点的邻居点的m属性赋给该点的feat_src_fc属性
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
            rst = graph.dstdata["feat_src_fc"]   # 把aggregate之后的点的feature取出来，之后准备过NGNN的MLP

            if self._use_symmetric_norm:
                degs = graph.dstdata["deg"]
                # degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # 这一段为NGNN加的MLP
            rst = rst.reshape(rst.shape[0],-1)
            for ngnn_layer in self.ngnn_layers:
                if isinstance(ngnn_layer, FMoEMLP):
                    rst, gate = ngnn_layer(rst, graph)
                    gate_std_ngnn = torch.std(gate, 1).mean()
                else:
                    rst = ngnn_layer(rst)
                rst = F.relu(rst)
            rst = rst.clone()
            rst = rst.reshape(rst.shape[0], self._n_heads, -1)

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            gate_std_attn_src = torch.tensor([1.0])     # 瞎造的一个
            return rst, gate_std_attn_src, gate_std_ngnn         
