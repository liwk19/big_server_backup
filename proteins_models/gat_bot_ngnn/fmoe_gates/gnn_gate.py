from .base_gate import BaseGate
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.ops import edge_softmax


class GCNGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.fc = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.expert_drop = expert_drop

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            degs_src = graph.out_degrees().float()  # 获取节点的度
            norm_src = torch.pow(degs_src, -0.5)
            norm_src[torch.isinf(norm_src)] = 0
            norm_src = norm_src.unsqueeze(1)
            graph.srcdata['feat_src_fc_gate'] = norm_src * graph.srcdata['feat_src_fc_gate']
            degs_dst = graph.in_degrees().float()  # 获取节点的度
            norm_dst = torch.pow(degs_dst, -0.5)
            norm_dst[torch.isinf(norm_dst)] = 0
            graph.dstdata['norm'] = norm_dst.unsqueeze(1)
            graph.update_all(message_func=fn.u_mul_v('feat_src_fc_gate', 'norm', 'm'), reduce_func=fn.sum('m', 'gate_src'))
            gate_src = graph.dstdata['gate_src']
        
        gate = self.fc(gate_src)
        gate = F.dropout(gate, p=self.expert_drop)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)        # 最高k个门的得分
        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)          # softmax一下得分

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class SageGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.fc = nn.Linear(2 * d_model, self.tot_expert)
        self.top_k = top_k
        self.expert_drop = expert_drop

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            graph.update_all(message_func=fn.copy_u('feat_src_fc_gate', 'm'), reduce_func=fn.mean('m', 'gate_src'))
            gate_src = graph.dstdata['gate_src']
        
        h_total = torch.cat([inp, gate_src], dim=1)
        gate = self.fc(h_total)
        
        gate = F.dropout(gate, p=self.expert_drop)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)        # 最高k个门的得分
        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)          # softmax一下得分

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class Sage2Gate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.fc = nn.Linear(3 * d_model, self.tot_expert)
        self.top_k = top_k
        self.expert_drop = expert_drop

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            graph.update_all(message_func=fn.copy_u('feat_src_fc_gate', 'm'), reduce_func=fn.mean('m', 'gate_src'))
            gate_src = graph.dstdata['gate_src']
        
        gate_dst = graph.dstdata['feat_dst_fc_gate']
        h_total = torch.cat([inp, gate_src, gate_dst], dim=1)
        gate = self.fc(h_total)
        
        gate = F.dropout(gate, p=self.expert_drop)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)        # 最高k个门的得分
        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)          # softmax一下得分

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class GATGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.src_fc = nn.Linear(d_model, self.tot_expert)
        self.dst_fc = nn.Linear(d_model, self.tot_expert)
        self.attn_src = nn.Linear(self.tot_expert, 1)
        self.attn_dst = nn.Linear(self.tot_expert, 1)
        self.top_k = top_k
        self.expert_drop = expert_drop
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            feat_src = graph.srcdata['feat_src_fc_gate']
            src_gate = self.src_fc(feat_src)
            dst_gate = self.dst_fc(inp)
            attn_src_ = self.attn_src(src_gate)
            attn_dst_ = self.attn_dst(dst_gate)
            graph.srcdata.update({'feat_src_fc_gate_': src_gate})
            graph.srcdata.update({"attn_src_gate": attn_src_})
            graph.dstdata.update({"attn_dst_gate": attn_dst_})
            graph.apply_edges(fn.u_add_v("attn_src_gate", "attn_dst_gate", "attn_node_gate"))
            e = graph.edata["attn_node_gate"]
            e = self.leaky_relu(e)
            graph.edata["a_gate"] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e("feat_src_fc_gate_", "a_gate", "m_gate"), fn.sum("m_gate", "gate"))
            gate = graph.dstdata['gate']
        
        gate = F.dropout(gate, p=self.expert_drop)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)        # 最高k个门的得分
        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)          # softmax一下得分

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class GAT2Gate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.src_fc = nn.Linear(d_model, self.tot_expert)
        self.dst_fc = nn.Linear(d_model, self.tot_expert)
        self.attn = nn.Linear(self.tot_expert, 1)
        self.top_k = top_k
        self.expert_drop = expert_drop
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            feat_src = graph.srcdata['feat_src_fc_gate']
            src_gate = self.src_fc(feat_src)
            dst_gate = self.dst_fc(inp)
            graph.srcdata.update({'feat_src_gate_': src_gate})
            graph.dstdata.update({"feat_dst_gate_": dst_gate})
            graph.apply_edges(fn.u_add_v("feat_src_gate_", "feat_dst_gate_", "attn_node_gate"))
            e = graph.edata["attn_node_gate"]
            e = self.leaky_relu(e)
            e = self.attn(e)
            graph.edata["a_gate"] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e("feat_src_gate_", "a_gate", "m_gate"), fn.sum("m_gate", "gate"))
            gate = graph.dstdata['gate']
        
        gate = F.dropout(gate, p=self.expert_drop)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)        # 最高k个门的得分
        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)          # softmax一下得分

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
