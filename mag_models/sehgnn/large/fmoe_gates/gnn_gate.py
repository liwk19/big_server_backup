r"""
GCN gate
"""
from .base_gate import BaseGate
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


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
            graph.ndata['_gate_src'] = norm_src * graph.ndata['_gate_src']
            degs_dst = graph.in_degrees().float()  # 获取节点的度
            norm_dst = torch.pow(degs_dst, -0.5)
            norm_dst[torch.isinf(norm_dst)] = 0
            graph.ndata['norm'] = norm_dst.unsqueeze(1)
            graph.update_all(message_func=fn.u_mul_v('_gate_src', 'norm', 'm'), reduce_func=fn.sum('m', 'gate_src'))
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
            # update_all is a message passing API.
            graph.update_all(message_func=fn.copy_u('_gate_src', 'm'), reduce_func=fn.mean('m', 'gate_src'))
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


class GATGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.fc = nn.Linear(2 * d_model, self.tot_expert)
        self.top_k = top_k
        self.expert_drop = expert_drop

    def forward(self, inp, graph, return_all_scores=False):
        with graph.local_scope():
            # update_all is a message passing API.
            graph.update_all(message_func=fn.copy_u('_gate_src', 'm'), reduce_func=fn.mean('m', 'gate_src'))
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
