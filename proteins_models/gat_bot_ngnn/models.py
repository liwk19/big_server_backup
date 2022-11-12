import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init
from fmoe_image import FMoE, FMoELinear   # fmoe_image是我写的一个拷贝过来的fmoe，用来方便我输出fmoe内部信息
from gat2 import GAT2Conv
import copy


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
    
    def pseudo_forward(self, inp, graph=None):
        output, expert = super().pseudo_forward(inp, graph)    # positionwise feed-forward
        output = self.dropout(output)
        return output, expert


# GAT卷积层
class GATConv(nn.Module):
    def __init__(self, node_feats, edge_feats, out_feats, n_heads=1, attn_drop=0.0, edge_drop=0.0,
        negative_slope=0.2, residual=True, activation=None, use_attn_dst=True,
        allow_zero_in_degree=True, use_symmetric_norm=False, num_expert=4, top_k=2,
        n_hidden_layers=2, fmoe2=0, moe_drp=0, gate='naive', moe_widget=['ngnn'], fmoe22=0, expert_drop=0
    ):
        super(GATConv, self).__init__()
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
        if 'attn_src_fc' in moe_widget:
            self.attn_src_fc = FMoEMLP(self._in_src_feats, n_heads, num_expert, 
                top_k, 0, moe_drp, gate='naive', expert_drop=expert_drop)         # 不支持fmoe2
        else:
            self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)  # attn_fc之后获得的为n_heads维度的
        if use_attn_dst:   # default is this
            if 'attn_dst_fc' in moe_widget:
                self.attn_dst_fc = FMoEMLP(self._in_src_feats, n_heads, num_expert, 
                    top_k, 0, moe_drp, 'naive', bias=False, expert_drop=expert_drop)     # 不支持fmoe2
            else:
                self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None
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

        if 'attn_src_fc' not in self.moe_widget:
            nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            if 'attn_dst_fc' not in self.moe_widget:
                nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            if 'attn_edge_fc' not in self.moe_widget:
                nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

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

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc})
            graph.srcdata.update({"feat_src_fc_gate": feat_src_fc.view(-1, self._n_heads * self._out_feats)})
            graph.dstdata.update({"feat_dst_fc_gate": feat_dst_fc.view(-1, self._n_heads * self._out_feats)})
            if isinstance(self.attn_src_fc, FMoEMLP):
                attn_src, gate = self.attn_src_fc(feat_src)
                attn_src = attn_src.view(-1, self._n_heads, 1)
                gate_std_attn_src = torch.std(gate, 1).mean()
            else:
                attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)
                gate_std_attn_src = torch.tensor([1.0])   # 造一个出来
            graph.srcdata.update({"attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                # apply_edges对所有边进行操作，生成边的特征。一个边从srcnode指向dstnode，
                # 把srcnode的'sttn_src'字段和dstnode的'attn_dst'字段相加，存到边的'attn_node'字段
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))  # 两个相加得到attn_node
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:   # 我们是这种
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)  # 输出维度为n_heads
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]  # e就是边本身的attn和边连接的两点的attention之和
            e = self.leaky_relu(e)

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
                    gate_std_ngnn = torch.tensor([1.0], device=rst.device)
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
            return rst, gate_std_attn_src, gate_std_ngnn
    
    def pseudo_forward(self, graph, feat_src, feat_edge=None):
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

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc})
            graph.srcdata.update({"feat_src_fc_gate": feat_src_fc.view(-1, self._n_heads * self._out_feats)})
            graph.dstdata.update({"feat_dst_fc_gate": feat_dst_fc.view(-1, self._n_heads * self._out_feats)})

            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)
            gate_std_attn_src = torch.tensor([1.0])   # 造一个出来
            graph.srcdata.update({"attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                # apply_edges对所有边进行操作，生成边的特征。一个边从srcnode指向dstnode，
                # 把srcnode的'sttn_src'字段和dstnode的'attn_dst'字段相加，存到边的'attn_node'字段
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))  # 两个相加得到attn_node
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:   # 我们是这种
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)  # 输出维度为n_heads
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]  # e就是边本身的attn和边连接的两点的attention之和
            e = self.leaky_relu(e)

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
            moe_in = copy.deepcopy(rst)
            for ngnn_layer in self.ngnn_layers:
                assert isinstance(ngnn_layer, FMoEMLP)
                rst, experts = ngnn_layer.pseudo_forward(rst, graph)
                moe_out = copy.deepcopy(rst)
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
            return rst, experts, moe_in, moe_out


# 这个类是GAT的主类了
class GAT(nn.Module):
    def __init__(self, node_feats, edge_feats, n_classes, n_layers, n_heads, n_hidden, edge_emb, 
        activation, dropout, input_drop, attn_drop, edge_drop, use_attn_dst=True, 
        allow_zero_in_degree=False, num_expert=4, top_k=2, fmoe2=0, 
        pred_fmoe=False, moe_drp=0, gate='naive', moe_widget=['ngnn'], fmoe22=0, expert_drop=0, gat2=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()  # 一堆GATConv
        self.norms = nn.ModuleList()  # 一堆batch normalization
        if 'encoder' in moe_widget:
            self.node_encoder = FMoEMLP(node_feats, n_hidden, num_expert, top_k, 0, moe_drp, 'naive', expert_drop=expert_drop)
        else:
            self.node_encoder = nn.Linear(node_feats, n_hidden)  # 初始把node特征转化到n_hidden维度
        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()  # 初始把edge特征给embed的一个模块

        for i in range(n_layers):   # 每层弄一次
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_hidden   # 这是多注意力机制
            if edge_emb > 0:   # edge_emb在外面传进来的是16
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))   # edge_encoder把边向量变成16维
            if gat2:
                self.convs.append(GAT2Conv(in_hidden, edge_emb, out_hidden, n_heads, attn_drop, edge_drop, 
                    use_attn_dst=use_attn_dst, allow_zero_in_degree=allow_zero_in_degree, use_symmetric_norm=False,
                    num_expert=num_expert, top_k=top_k, fmoe2=fmoe2, 
                    moe_drp=moe_drp, gate=gate, moe_widget=moe_widget, fmoe22=fmoe22, expert_drop=expert_drop))
            else:
                self.convs.append(GATConv(in_hidden, edge_emb, out_hidden, n_heads, attn_drop, edge_drop, 
                    use_attn_dst=use_attn_dst, allow_zero_in_degree=allow_zero_in_degree, use_symmetric_norm=False,
                    num_expert=num_expert, top_k=top_k, fmoe2=fmoe2, 
                    moe_drp=moe_drp, gate=gate, moe_widget=moe_widget, fmoe22=fmoe22, expert_drop=expert_drop))
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        if pred_fmoe:  #最后的分类头（pred是predict含义）
            self.pred_linear = FMoEMLP(n_heads * n_hidden, n_classes, num_expert, top_k)
        else:
            self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes) 
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.name = 'lyr{} hed{} hid{} drp{} idrp{} edrp{} mdrp{} '.format(n_layers, n_heads, n_hidden, float(dropout), float(input_drop), float(edge_drop), float(moe_drp))
        self.name += 'ept{} k{} hly{} moe{} pmoe{}'.format(num_expert, top_k, 1, fmoe2, int(pred_fmoe))
    
    def __str__(self):
        return self.name

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g   # this case, g is a list of subgraphs
        h = subgraphs[0].srcdata["feat"]
        h = self.node_encoder(h)      # 先encode到n_hidden维度
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)          # 然后input_drop一下

        h_last = None   # h_last为上一次的h，初始化为None
        gate_std_attn_src_means = None
        gate_std_ngnn_means = None

        for i in range(self.n_layers):    # 遍历各层，每层使用的图是subgraphs[i]
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)   # 把edge的特征encode到edge_emb维
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None
            h, gate_std_attn_src, gate_std_ngnn = self.convs[i](subgraphs[i], h, efeat_emb)  # 做GAT卷积
            h = h.flatten(1, -1) 
            
            gate_std_attn_src_mean = gate_std_attn_src.unsqueeze(0)
            if gate_std_attn_src_means == None:
                gate_std_attn_src_means = gate_std_attn_src_mean
            else:
                gate_std_attn_src_means = torch.concat((gate_std_attn_src_means, gate_std_attn_src_mean))
            gate_std_ngnn_mean = gate_std_ngnn.unsqueeze(0)
            if gate_std_ngnn_means == None:
                gate_std_ngnn_means = gate_std_ngnn_mean
            else:
                gate_std_ngnn_means = torch.concat((gate_std_ngnn_means, gate_std_ngnn_mean))

            if h_last is not None:
                h += h_last[: h.shape[0], :]   # skip connection
            h_last = h
            h = self.norms[i](h)   # skip connection之后依次过norm，activation和dropout
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)   # 最后分类头做分类
        return h, gate_std_attn_src_means.mean(), gate_std_ngnn_means.mean()
    
    def pseudo_forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g   # this case, g is a list of subgraphs
        h = subgraphs[0].srcdata["feat"]
        h = self.node_encoder(h)      # 先encode到n_hidden维度
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)          # 然后input_drop一下

        h_last = None   # h_last为上一次的h，初始化为None
        experts_list, moe_in_list, moe_out_list = [], [], []

        for i in range(self.n_layers):    # 遍历各层，每层使用的图是subgraphs[i]
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)   # 把edge的特征encode到edge_emb维
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None
            h, expert, moe_in, moe_out = self.convs[i].pseudo_forward(subgraphs[i], h, efeat_emb)  # 做GAT卷积
            experts_list.append(expert)
            moe_in_list.append(moe_in)
            moe_out_list.append(moe_out)
            h = h.flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]   # skip connection
            h_last = h
            h = self.norms[i](h)   # skip connection之后依次过norm，activation和dropout
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        return experts_list, moe_in_list, moe_out_list
