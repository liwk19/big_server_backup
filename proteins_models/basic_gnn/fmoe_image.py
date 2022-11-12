r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.autograd import Function
import math
import fmoe_cuda
import tree
import os
import torch
import torch.nn as nn
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NoisyGate, SwipeGate, SwitchGate
from fmoe_gates.naive_gate import NaiveGate
from fmoe_gates.mlp_gate import MLPGate
from fmoe_gates.gnn_gate import GCNGate, SageGate
from fmoe.fastermoe.config import switch_from_env
import fmoe_cuda
import torch.distributed as dist


# 下面两个函数来自fmoe.functions
def count_by_gate(gate, num_expert, world_size, require_pos=True):
    with torch.no_grad():
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.int32
        )
        fmoe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()    # 这里统计分配给每个expert的点数。是一个长度为num_expert * world_size的tensor

        if world_size > 1:
            global_expert_count = fmoe_cuda.expert_exchange(     # 这里报错！
                local_expert_count, num_expert, world_size
            )
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            fmoe_cuda.assign_pos(lec_cum, gate, pos)
    return pos, local_expert_count, global_expert_count


def prepare_forward(gate, num_expert, world_size):
    r"""
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
    """
    pos, local_expert_count, global_expert_count = count_by_gate(gate, 
            num_expert, world_size)
    with torch.no_grad():
        fwd_expert_count = global_expert_count.view(world_size,
                num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count.cpu(),
        global_expert_count.cpu(),
        fwd_expert_count.cpu(),
        fwd_batch_size,
    )


# pylint: disable=broad-except
# pylint: disable=protected-access
def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")


def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    global _moe_group
    _moe_group = comm
    fmoe_cuda.ensure_nccl(comm, t)


def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)       # 这里出了问题
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )
    x = tree.map_structure(scatter_func, inp)

    x = expert_fn(x, fwd_expert_count)
    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )

    outp = tree.map_structure(gather_func, x)
    return outp


fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    print('wow!')
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None, # (this param is deprecated)
        slice_group=None, moe_group=None, top_k=2, gate='naive', expert=None,
        gate_hook=None, mask=None, mask_dict=None, expert_drop=0):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        if gate == 'naive':
            self.gate = NaiveGate(d_model, num_expert, world_size, top_k, expert_drop)
        elif gate == 'noisy':
            self.gate = NoisyGate(d_model, num_expert, world_size, top_k)
        elif gate == 'swipe':
            self.gate = SwipeGate(d_model, num_expert, world_size, top_k)
        elif gate == 'switch':
            self.gate = SwitchGate(d_model, num_expert, world_size, top_k)
        elif gate == 'mlp':
            self.gate = MLPGate(d_model, num_expert, world_size, top_k, expert_drop)
        elif gate == 'gcn':
            self.gate = GCNGate(d_model, num_expert, world_size, top_k, expert_drop)
        elif gate == 'sage':
            self.gate = SageGate(d_model, num_expert, world_size, top_k, expert_drop)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group
        self.expert_drop = nn.Dropout(expert_drop)

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp, graph=None):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(     # moe_inp.shape is [数据点数，维度数]
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )   # moe_inp_batch_size是[数据点数]（一个列表，里面只有这一个元素）
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)   # 这句话报错
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_inp = tree.map_structure(slice_func, moe_inp)
        
        # gate_top_k_idx是[点数，4]，4为4个experts的idx
        if isinstance(self.gate, GCNGate) or isinstance(self.gate, SageGate):
            gate_top_k_idx, gate_score = self.gate(moe_inp, graph)    
        else:
            gate_top_k_idx, gate_score, gate = self.gate(moe_inp, return_all_scores=True)
        # 分数从高到低，gate_score形状也是[点数，4]，4个值分别为4个分数，相加为1。
        # gate_score = self.expert_drop(gate_score)
        # gate_score_sum = torch.sum(gate_score, 1)
        # gate_score_sum = torch.unsqueeze(gate_score_sum, 1)
        # gate_score = gate_score / gate_score_sum

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors  默认为None
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]
        
        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )     # fwd.shape是[2*点数，维数]，那个2对应expert_num

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)

        else:
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)   # moe_outp的形状是[点数，2，维数]
        
        gate_score = gate_score.view(-1, 1, self.top_k)   # gate_score本来是[点数，2]，现变成[点数，1，2]

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)   # moe_outp为加权去和后的结果，为[点数，维数]

        if self.slice_size > 1:    # 默认为1
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )       # 为[点数]（一个列表，仅此一个元素
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, gate
    

    def pseudo_forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(     # moe_inp.shape is [数据点数，维度数]
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )   # moe_inp_batch_size是[数据点数]（一个列表，里面只有这一个元素）
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_inp = tree.map_structure(slice_func, moe_inp)
        
        gate_top_k_idx, gate_score = self.gate(moe_inp)    # gate_top_k_idx是[点数，4]，4为4个experts的idx
        # 分数从高到低，gate_score形状也是[点数，4]，4个值分别为4个分数，相加为1。

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors  默认为None
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]
        
        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )     # fwd.shape是[2*点数，维数]，那个2对应expert_num
        record_fwd = fwd

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)

        else:
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)   # moe_outp的形状是[点数，2，维数]
        
        gate_score = gate_score.view(-1, 1, self.top_k)   # gate_score本来是[点数，2]，现变成[点数，1，2]

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)   # moe_outp为加权去和后的结果，为[点数，维数]

        if self.slice_size > 1:    # 默认为1
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )       # 为[点数]（一个列表，仅此一个元素
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, gate_top_k_idx, record_fwd


class MOELinear(Function):
    r"""
    Computes linear operators within one GPU on different experts simutaneously.
    """

    @staticmethod   # 为静态方法，可以直接用“类名.方法名”调用
    def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
        global_output_buf = fmoe_cuda.linear_forward(
            global_input_buf, fwd_expert_count, weight, bias
        )
        variables = (global_input_buf, fwd_expert_count, weight, bias)
        ctx.save_for_backward(*variables)
        return global_output_buf

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, fwd_expert_count, weight, bias) = ctx.saved_tensors
        grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.linear_backward(
            grad_out, input_buf, fwd_expert_count, weight, bias
        )

        if not torch.is_tensor(bias):
            grad_bias = None

        return grad_inp_buf, None, grad_weight, grad_bias



class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class _Expert2(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """
    def __init__(self, num_expert, d_input, d_hidden, d_output, activation, rank=0, bias=True):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_input, d_hidden, bias=bias, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_output, bias=bias, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class _Expert(nn.Module):
    r"""
    An expert using 1 FMoELinear module, instead of 2.
    """
    def __init__(self, num_expert, d_input, d_output, rank=0, bias=True):
        super().__init__()
        self.htoh = FMoELinear(num_expert, d_input, d_output, bias=bias, rank=rank)

    def forward(self, inp, fwd_expert_count):
        x = self.htoh(inp, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """
    def __init__(
        self, num_expert=32, d_input=1024, d_output=1024, top_k=2, activation=torch.nn.GELU(),
        expert_dp_comm="none", expert_rank=0, fmoe2=0, gate='naive', bias=True
    ):
        super().__init__(num_expert=num_expert, d_model=d_input, top_k=top_k, world_size=1, gate=gate)
        if fmoe2 > 0:
            self.experts = _Expert2(num_expert, d_input, fmoe2, d_output, activation, rank=expert_rank, bias=bias)
        else:
            self.experts = _Expert(num_expert, d_input, d_output, rank=expert_rank, bias=bias)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp, graph=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        # original_shape = inp.shape
        # inp = inp.reshape(-1, self.d_model)     # 对于本模型来说这一步操作没改变任何东西
        output = super().forward(inp, graph)
        # return output.reshape(original_shape)
        return output
    
    def pseudo_forward(self, inp: torch.Tensor):
        # original_shape = inp.shape
        # inp = inp.reshape(-1, self.d_model)     # 对于本模型来说这一步操作没改变任何东西
        output, experts, record_fwd = super().pseudo_forward(inp)
        # return output.reshape(original_shape)
        return output, experts, record_fwd
