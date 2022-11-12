import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
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


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_expert, 
        top_k, expert_drop, gate):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            # self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(FMoEMLP(hidden_channels, hidden_channels, num_expert, top_k, 
                expert_drop=expert_drop, gate=gate))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            if not isinstance(lin, FMoEMLP):
                lin.reset_parameters()

    def forward(self, x):
        gate_std_means = None

        for lin in self.lins[0:-1]:
            if isinstance(lin, FMoEMLP):
                x, gate = lin(x)
                gate_std_mean = torch.std(gate, 1).mean()
                gate_std_mean = gate_std_mean.unsqueeze(0)
                if gate_std_means == None:
                    gate_std_means = gate_std_mean
                else:
                    gate_std_means = torch.concat((gate_std_means, gate_std_mean))
            else:
                x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lins[-1](x)
        return x, gate_std_means.mean()


def train(model, x, y_true, train_idx, optimizer, alpha=0):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out, gate_std_mean = model(x)
    out = out[train_idx]
    loss1 = criterion(out, y_true[train_idx].to(torch.float))
    loss2 = gate_std_mean
    loss = loss1 + alpha * loss2
    loss.backward()
    optimizer.step()
    return loss.item(), loss1.item(), loss2.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()
    y_pred, gate_std_mean = model(x)
    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']
    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--num_expert', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--expert_drop', type=float, default=0.0)
    parser.add_argument("--gate", type=str, default='mlp', help="FMOE's gate", 
        choices=['naive', 'mlp'])
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    if args.expert_drop > 0:
        assert args.top_k > 1, 'expert_drop only supports more than one selected experts'

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root='/data/liweikai/gat_bot_ngnn/dataset')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to('cpu')
    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)

    x = x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)
    model = MLP(x.size(-1) + 112, args.hidden_channels, 112, args.num_layers, args.dropout,
        args.num_expert, args.top_k, args.expert_drop, args.gate).to(device)
    print('#parameters:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))
    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        best_valid, final_test = 0, 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, 1 + args.epochs):
            loss, loss1, loss2 = train(model, x, y_true, train_idx, optimizer, args.alpha)

            if epoch % args.eval_steps == 0:
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)
                train_rocauc, valid_rocauc, test_rocauc = result
                if valid_rocauc > best_valid:
                    best_valid = valid_rocauc
                    final_test = test_rocauc

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}, '
                          f'Train: {train_rocauc:.4f}, '
                          f'Valid: {valid_rocauc:.4f}, '
                          f'Test: {test_rocauc:.4f}, '
                          f'Best Valid: {best_valid:.4f}, '
                          f'Final Test: {final_test:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()
    print('#parameters:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))


if __name__ == "__main__":
    main()
