import torch.nn as nn
import torch.nn.functional as F
from cogdl.layers import GCNLayer
from cogdl.utils import get_activation
import cogdl.datasets.plantoid_data.read_planetoid_data
from fmoe import FMoETransformerMLP


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, moe_num_expert=64, moe_top_k=2):
        activation = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k, activation=activation
        )

        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(d_model)

    def forward(self, inp):
        ##### positionwise feed-forward
        core_out = super().forward(inp)
        core_out = self.dropout(core_out)

        ##### residual connection + batch normalization
        output = self.bn_layer(inp + core_out)

        return output


class GraphConvBlock(nn.Module):
    def __init__(self, conv_func, conv_params, in_feats, out_feats, dropout=0.0, residual=False):
        super(GraphConvBlock, self).__init__()

        self.graph_conv = conv_func(**conv_params, in_features=in_feats, out_features=out_feats)
        self.pos_ff = CustomizedMoEPositionwiseFF(out_feats, out_feats * 2, dropout, moe_num_expert=64, moe_top_k=2)
        self.dropout = dropout
        if residual is True:
            assert in_feats is not None
            self.res_connection = nn.Linear(in_feats, out_feats)
        else:
            self.res_connection = None

    def reset_parameters(self):
        """Reinitialize model parameters."""
        # self.graph_conv.reset_parameters()
        if self.res_connection is not None:
            self.res_connection.reset_parameters()

    def forward(self, graph, feats):
        new_feats = self.graph_conv(graph, feats)
        if self.res_connection is not None:
            res = self.res_connection
            new_feats = new_feats + res
            new_feats = F.dropout(new_feats, p=self.dropout, training=self.training)

        new_feats = self.pos_ff(new_feats)

        return new_feats


class MoEGCN(nn.Module):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        in_features (int) : Number of input features.
        out_features (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    def __init__(
        self, in_feats, hidden_size, out_feats, num_layers=2, dropout=0.5, activation="relu", residual=True, norm=None
    ):
        super(MoEGCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * num_layers
        conv_func = GCNLayer
        conv_params = {
            "dropout": dropout,
            "norm": norm,
            "residual": residual,
            "activation": activation,
        }
        self.layers = nn.ModuleList(
            [
                GraphConvBlock(
                    conv_func,
                    conv_params,
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)
        self.final_cls = nn.Linear(hidden_size, out_feats)

    def embed(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.layers[i](graph, h)
        return h

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        h = self.final_cls(h)
        return h

    def predict(self, data):
        return self.forward(data)


# the following is written by me
dataset = read_planetoid_data('./data', 'Cora')
data = dataset[0]
print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MoEGCN(dataset.num_features, 128, dataset.num_classes)
data.to(device)
model.to(device)

lr = 0.01
wd = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
best_acc = 0
best_model = None
patience = 0
epoch = 0
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc
        else:
            if patience == max_patience:
                break
            else:
                patience = patience + 1
        print(f'Accuracy: {acc:.4f}')

print(epoch)
model = best_model
model.eval()
with torch.no_grad():
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
