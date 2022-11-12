import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.transforms as T
from torch_scatter import scatter_add, scatter
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, Size
from torch_geometric.utils import add_remaining_self_loops, to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple, Union
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

    def forward(self, inp):
        output, gate = super().forward(inp)
        output = self.dropout(output)
        return output, gate


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


# 改造自PyG的GCNConv
class MyGCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, 
        cached: bool = False, add_self_loops: bool = True, normalize: bool = True, 
        bias: bool = True, num_expert=16, top_k=1, expert_drop=0, gate='naive', **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        # self.lin = nn.Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.lin = FMoEMLP(in_channels, out_channels, num_expert, top_k, expert_drop=expert_drop, gate=gate)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.lin, nn.Linear):
            self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x, gate = self.lin(x)
        gate_std_mean = torch.std(gate, 1).mean()
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        return out, gate_std_mean

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


# 参考自PyG
class MySAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, num_expert, top_k,
        aggr: str = 'mean', normalize: bool = False, root_weight: bool = True,
        project: bool = False, bias: bool = True, expert_drop=0,
        gate='naive', **kwargs
    ):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        # self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_l = FMoEMLP(in_channels[0], out_channels, num_expert, top_k, expert_drop=expert_drop,
            gate=gate)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        if self.aggr is None:
            self.lstm.reset_parameters()
        if not isinstance(self.lin_l, FMoEMLP):
            self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out, gate = self.lin_l(out)
        gate_std_mean = torch.std(gate, 1).mean()

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, gate_std_mean

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.aggr is not None:
            return scatter(x, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

        # LSTM aggregation:
        if ptr is None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(f"Can not utilize LSTM-style aggregation inside "
                             f"'{self.__class__.__name__}' in case the "
                             f"'edge_index' tensor is not sorted by columns. "
                             f"Run 'sort_edge_index(..., sort_by_row=False)' "
                             f"in a pre-processing step.")

        x, mask = to_dense_batch(x, batch=index, batch_size=dim_size)
        out, _ = self.lstm(x)
        return out[:, -1]

    def __repr__(self) -> str:
        aggr = self.aggr if self.aggr is not None else 'lstm'
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={aggr})')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, 
        num_expert, top_k, expert_drop, gate, ngnn, ngnn_num_expert, ngnn_top_k,
        ngnn_gate, ngnn_expert_drop):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.convs.append(MyGCNConv(in_channels, hidden_channels, normalize=False, 
            num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate))
        for _ in range(num_layers - 2):
            # self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.convs.append(MyGCNConv(hidden_channels, hidden_channels, normalize=False, 
                num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate))

        self.dropout = dropout
        self.ngnn = ngnn
        if ngnn == 0:
            self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        if ngnn > 0:
            self.convs.append(MyGCNConv(hidden_channels, ngnn, normalize=False, 
                num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate))
            self.ngnn_layer = FMoEMLP(ngnn, out_channels, num_expert=ngnn_num_expert, 
                top_k=ngnn_top_k, gate=ngnn_gate, expert_drop=ngnn_expert_drop)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        gate_std_means = None
        gate_std_ngnn = None
        if self.ngnn:
            for conv in self.convs:
                if isinstance(conv, MyGCNConv):
                    x, gate_std_mean = conv(x, adj_t)
                    gate_std_mean = gate_std_mean.unsqueeze(0)
                    if gate_std_means == None:
                        gate_std_means = gate_std_mean
                    else:
                        gate_std_means = torch.concat((gate_std_means, gate_std_mean))
                else:
                    x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x, gate_std_ngnn = self.ngnn_layer(x)
        
        else:
            for conv in self.convs[:-1]:
                if isinstance(conv, MyGCNConv):
                    x, gate_std_mean = conv(x, adj_t)
                    gate_std_mean = gate_std_mean.unsqueeze(0)
                    if gate_std_means == None:
                        gate_std_means = gate_std_mean
                    else:
                        gate_std_means = torch.concat((gate_std_means, gate_std_mean))
                else:
                    x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)
        
        if gate_std_means == None:
            gate_std_means = torch.tensor([1.])
        if gate_std_ngnn == None:
            gate_std_ngnn = torch.tensor([1.])
        return x, gate_std_means.mean(), gate_std_ngnn.mean()


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_expert, top_k, gate, expert_drop, ngnn, ngnn_num_expert, ngnn_top_k,
                 ngnn_gate, ngnn_expert_drop):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        # self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(MySAGEConv(in_channels, hidden_channels, 
                num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate))
        for _ in range(num_layers - 2):
            # self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(MySAGEConv(hidden_channels, hidden_channels, 
                num_expert=num_expert, top_k=top_k, expert_drop=expert_drop, gate=gate))
        if ngnn == 0:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        else:
            self.convs.append(SAGEConv(hidden_channels, ngnn))
        self.dropout = dropout

        self.ngnn = ngnn
        if ngnn > 0:
            self.ngnn_layer = FMoEMLP(ngnn, out_channels, num_expert=ngnn_num_expert, 
                top_k=ngnn_top_k, gate=ngnn_gate, expert_drop=ngnn_expert_drop)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        gate_std_means = None
        gate_std_ngnn = None
        for conv in self.convs[:-1]:
            if isinstance(conv, MySAGEConv):
                x, gate_std_mean = conv(x, adj_t)
                gate_std_mean = gate_std_mean.unsqueeze(0)
                if gate_std_means == None:
                    gate_std_means = gate_std_mean
                else:
                    gate_std_means = torch.concat((gate_std_means, gate_std_mean))
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        if self.ngnn:
            x, gate_std_ngnn = self.ngnn_layer(x)
            return x, gate_std_means.mean(), gate_std_ngnn.mean()
        return x, gate_std_means.mean(), None


def train(model, data, train_idx, optimizer, alpha=0, ngnn_alpha=0):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out, gate_std_mean, gate_std_ngnn = model(data.new_x, data.adj_t)
    out = out[train_idx]
    loss1 = criterion(out, data.y[train_idx].to(torch.float))
    loss2 = gate_std_mean
    loss3 = gate_std_ngnn
    if alpha == 0 and ngnn_alpha == 0:
        loss = loss1
    elif alpha == 0 and ngnn_alpha != 0:
        loss = loss1 + ngnn_alpha * loss3
    elif alpha != 0 and ngnn_alpha == 0:
        loss = loss1 + alpha * loss2
    elif alpha != 0 and ngnn_alpha != 0:
        loss = loss1 + alpha * loss2 + ngnn_alpha * loss3
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, label_weight):
    model.eval()
    data, _ = add_labels(data, train_idx=split_idx['train'], label_rate=1.0, label_weight=label_weight)
    y_pred, gate_std_mean, gate_std_ngnn = model(data.new_x, data.adj_t)
    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']
    return train_rocauc, valid_rocauc, test_rocauc


def add_labels(data, train_idx, label_rate, label_weight=1):
    mask = torch.rand(train_idx.shape) < label_rate
    train_labels_idx = train_idx[mask]    
    train_pred_idx = train_idx[~mask]
    added_label = torch.zeros_like(data.y)
    added_label[train_labels_idx] = data.y[train_labels_idx]
    data.new_x = torch.concat((data.x, added_label), 1)
    return data, train_pred_idx


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--num_expert', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--expert_drop', type=float, default=0.4)
    parser.add_argument("--gate", type=str, default='mlp', help="FMOE's gate", 
        choices=['naive', 'mlp', 'gcn', 'sage', 'gat'])
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--ngnn", type=int, default=0)  # 0 means no ngnn
    parser.add_argument('--ngnn_num_expert', type=int, default=2)
    parser.add_argument('--ngnn_top_k', type=int, default=2)
    parser.add_argument('--ngnn_expert_drop', type=float, default=0.4)
    parser.add_argument("--ngnn_gate", type=str, default='mlp', help="FMOE's gate", 
        choices=['naive', 'mlp', 'gcn', 'sage', 'gat'])
    parser.add_argument("--ngnn_alpha", type=float, default=0.0)
    parser.add_argument("--label_rate", type=float, default=0.5)
    parser.add_argument("--label_warmup", type=int, nargs='+', default=[3010, 3010])
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygNodePropPredDataset(name='ogbn-proteins', 
        root='/data/liweikai/gat_bot_ngnn/dataset', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features + 112, args.hidden_channels, 112, args.num_layers, args.dropout, 
            num_expert=args.num_expert, top_k=args.top_k, expert_drop=args.expert_drop, gate=args.gate,
            ngnn=args.ngnn, ngnn_num_expert=args.ngnn_num_expert, ngnn_top_k=args.ngnn_top_k, 
            ngnn_expert_drop=args.ngnn_expert_drop, ngnn_gate=args.ngnn_gate).to(device)
    else:
        model = GCN(data.num_features + 112, args.hidden_channels, 112, args.num_layers, args.dropout,
            num_expert=args.num_expert, top_k=args.top_k, expert_drop=args.expert_drop, gate=args.gate,
            ngnn=args.ngnn, ngnn_num_expert=args.ngnn_num_expert, ngnn_top_k=args.ngnn_top_k, 
            ngnn_expert_drop=args.ngnn_expert_drop, ngnn_gate=args.ngnn_gate).to(device)
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    print('#parameters:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))
    data = data.to(device)
    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        best_valid, final_test, best_test, best_epoch = 0, 0, 0, 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, 1 + args.epochs):
            if epoch < args.label_warmup[0]:
                label_weight = 0.0
            elif epoch > args.label_warmup[1]:
                label_weight = 1
            else:
                label_weight = (epoch - args.label_warmup[0]) / (args.label_warmup[1] - args.label_warmup[0])
            data, train_pred_idx = add_labels(data, train_idx, label_rate=args.label_rate, label_weight=label_weight)
            loss = train(model, data, train_idx, optimizer, args.alpha, args.ngnn_alpha)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator, label_weight=label_weight)
                logger.add_result(run, result)
                train_rocauc, valid_rocauc, test_rocauc = result
                if valid_rocauc > best_valid:
                    best_valid = valid_rocauc
                    final_test = test_rocauc
                
                if test_rocauc > best_test:
                    best_test = test_rocauc
                    best_epoch = epoch

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_rocauc:.4f}, '
                          f'Valid: {valid_rocauc:.4f}, '
                          f'Test: {test_rocauc:.4f}, '
                          f'Best Valid: {best_valid:.4f}, '
                          f'Final Test: {final_test:.4f}')

        logger.print_statistics(run)
        print(f'Best Test: {best_test:.4f}, ' f'Best Test Epoch: {best_epoch:.4f}')

    logger.print_statistics()
    print('#parameters:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))


if __name__ == "__main__":
    main()
