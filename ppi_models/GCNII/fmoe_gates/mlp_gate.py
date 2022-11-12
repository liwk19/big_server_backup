r"""
Naive gate
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k, expert_drop):
        super().__init__(num_expert, world_size)
        self.fc1 = nn.Linear(d_model, 16)
        self.fc2 = nn.Linear(16, self.tot_expert)
        self.top_k = top_k
        self.expert_drop = expert_drop

    def forward(self, inp, return_all_scores=False):
        h = self.fc1(inp)
        h = F.relu(h)
        gate = self.fc2(h)
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
