from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from utils import *
from model import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid
import warnings
import time


warnings.filterwarnings('ignore')       # 关掉对torch.range的warning，太烦人了

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=9, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=2000, help='Patience')
parser.add_argument('--data', default='ppi', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument("--num_expert", type=int, default=1, help="number of experts")
parser.add_argument("--top_k", type=int, default=1, help="selected number of experts")
parser.add_argument("--gate", type=str, default='naive', help="FMOE's gate", choices=['naive', 'mlp'])
parser.add_argument("--moe_widget", type=str, nargs='+', default=[], choices=['ngnn'])
parser.add_argument('--label_rate', type=float, default=0.5)   # label used as feature rate
parser.add_argument("--label_warmup", type=float, nargs='+', default=[11000, 11000])
parser.add_argument('--label_onehot', action='store_true', default=False)
parser.add_argument('--expert_drop', type=float, default=0.0)
parser.add_argument('--reload', type=str, default='')       # reload a checkpt_file and continue training
parser.add_argument('--start_epoch', type=int, default=0)
args = parser.parse_args()

# setup seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

# Load data
train_adj, val_adj, test_adj, train_feat, val_feat, test_feat, train_labels, val_labels, \
    test_labels, train_nodes, val_nodes, test_nodes = load_ppi()
train_adj = [i.to(device) for i in train_adj]
val_adj = [i.to(device) for i in val_adj]
test_adj = [i.to(device) for i in test_adj]
train_feat = [i.to(device) for i in train_feat]
val_feat = [i.to(device) for i in val_feat]
test_feat = [i.to(device) for i in test_feat]
train_labels = [i.to(device) for i in train_labels]
val_labels = [i.to(device) for i in val_labels]
test_labels = [i.to(device) for i in test_labels]

checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print('checkpt_file:', checkpt_file)

n_feat = train_feat[0].shape[1]
if args.label_warmup[0] < args.epochs:
    if args.label_onehot:
        n_feat = n_feat + 2 * train_labels[0].shape[1]
    else:
        n_feat = n_feat + train_labels[0].shape[1]

model = GCNIIppi(nfeat=n_feat,
    nlayers=args.layer,
    nhidden=args.hidden,
    nclass=train_labels[0].shape[1],
    dropout=args.dropout,
    lamda = args.lamda, 
    alpha=args.alpha,
    variant=args.variant,
    num_expert=args.num_expert,
    top_k=args.top_k,
    gate=args.gate,
    moe_widget=args.moe_widget,
    expert_drop=args.expert_drop).to(device)

print('parameter number:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
if len(args.reload):
    model.load_state_dict(torch.load(args.reload))
    print('reload model:', args.reload)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
loss_fcn = torch.nn.BCELoss()


# adapted from DGL
def evaluate(feats, model, idx, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        output = model(feats, subgraph)
        loss_data = loss_fcn(output[:idx], labels[:idx].float())
        predict = np.where(output[:idx].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels[:idx].data.cpu().numpy(), predict, average='micro')
        return score, loss_data.item()


idx = torch.LongTensor(range(20))        # 分为20个batch
loader = Data.DataLoader(dataset=idx, batch_size=1, shuffle=True, num_workers=0)


def train(label_weight, if_add_label):
    model.train()
    loss_tra = 0
    for step, batch in enumerate(loader):  
        # print(step)     # 共20个step (因为训练集是20张图)
        # train_nodes[batch]保存了这次子图里的真实点数，后面的点都是“虚无”的，它们的label和feature都是0
        # batch_adj = train_adj[batch[0]].to(device)
        # batch_feature = train_feat[batch[0]].to(device)
        # batch_label = train_labels[batch[0]].to(device)
        batch_adj = train_adj[batch[0]]
        batch_feature = train_feat[batch[0]]
        batch_label = train_labels[batch[0]]
        
        if if_add_label:
            train_idx = torch.range(0, train_nodes[batch] - 1, dtype=int)
            mask = torch.rand(train_idx.shape[0]) < args.label_rate
            train_labels_idx = train_idx[mask]    # mask_rate是把train_label加入feature的概率
            train_pred_idx = train_idx[~mask]   # 没加入feature的就是要预测的

            if args.label_onehot:
                add_label = torch.zeros([batch_label.shape[0], 2 * batch_label.shape[1]], device=device, dtype=torch.float32)
                if train_nodes[batch] != batch_label.shape[0]:
                    batch_label_long = batch_label.long()
                    for i in range(batch_label.shape[1]):
                        add_label[train_labels_idx, 2 * i + batch_label_long[train_labels_idx, i]] = label_weight
            else:
                add_label = torch.zeros_like(batch_label)
                add_label[train_labels_idx] = batch_label[train_labels_idx] * label_weight
            batch_feature = torch.cat((batch_feature, add_label), 1)

        optimizer.zero_grad()
        output = model(batch_feature, batch_adj)
        if if_add_label:
            loss_train = loss_fcn(output[train_pred_idx], batch_label[train_pred_idx])
        else:
            loss_train = loss_fcn(output, batch_label)
        loss_train.backward() 
        optimizer.step()
        loss_tra += loss_train.item()
    loss_tra /= 20
    return loss_tra


def validation(label_weight, if_add_label):
    loss_val = 0
    acc_val = 0
    for batch in range(2):      # 分2个batch去跑验证集
        # batch_adj = val_adj[batch].to(device)
        # batch_feature = val_feat[batch].to(device)
        # batch_label = val_labels[batch].to(device)
        batch_adj = val_adj[batch]
        batch_feature = val_feat[batch]
        batch_label = val_labels[batch]

        if if_add_label:
            val_idx = torch.range(0, val_nodes[batch] - 1, dtype=int)
            if args.label_onehot:
                add_label = torch.zeros([batch_label.shape[0], 2 * batch_label.shape[1]], device=device, dtype=torch.float32)
                batch_label_long = batch_label.long()
                for i in range(batch_label.shape[1]):
                    add_label[val_idx, 2 * i + batch_label_long[val_idx, i]] = label_weight  # val时全用上
            else:
                add_label = torch.zeros_like(batch_label)
                add_label[val_idx] = batch_label[val_idx] * label_weight
            batch_feature = torch.cat((batch_feature, add_label), 1)

        score, val_loss = evaluate(batch_feature, model, val_nodes[batch], batch_adj, batch_label, loss_fcn)
        loss_val += val_loss
        acc_val += score
    loss_val /= 2
    acc_val /= 2
    return loss_val, acc_val


def test(label_weight, if_add_label):
    model.load_state_dict(torch.load(checkpt_file))
    loss_test = 0
    acc_test = 0
    for batch in range(2):     # 分2个batch去test
        # batch_adj = test_adj[batch].to(device)
        # batch_feature = test_feat[batch].to(device)
        # batch_label = test_labels[batch].to(device)
        batch_adj = test_adj[batch]
        batch_feature = test_feat[batch]
        batch_label = test_labels[batch]

        if if_add_label:
            test_idx = torch.range(0, test_nodes[batch] - 1, dtype=int)
            if args.label_onehot:
                add_label = torch.zeros([batch_label.shape[0], 2 * batch_label.shape[1]], device=device, dtype=torch.float32)
                batch_label_long = batch_label.long()
                for i in range(batch_label.shape[1]):
                    add_label[test_idx, 2 * i + batch_label_long[test_idx, i]] = label_weight  # val时全用上
            else:
                add_label = torch.zeros_like(batch_label)
                add_label[test_idx] = batch_label[test_idx] * label_weight
            batch_feature = torch.cat((batch_feature, add_label), 1)
        
        score, loss = evaluate(batch_feature, model, test_nodes[batch], batch_adj, batch_label, loss_fcn)
        loss_test += loss
        acc_test += score
    acc_test /= 2
    loss_test /= 2
    return acc_test


t_total = time.time()
bad_counter = 0
best_val = 0
best_epoch = 0
total_time = 0
if_add_label = args.label_warmup[0] < args.epochs

for epoch in range(args.start_epoch, args.epochs):
    tic = time.time()
    if epoch <= args.label_warmup[0]:
        label_weight = 0.0
    elif epoch >= args.label_warmup[1]:
        label_weight = 1.0
    else:
        label_weight = (epoch - args.label_warmup[0]) / (args.label_warmup[1] - args.label_warmup[0])
    loss_tra = train(label_weight, if_add_label)
    toc = time.time()
    total_time += toc - tic
    loss_val, acc_val = validation(label_weight, if_add_label)

    if epoch < 10:
        peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
        print('Peak memuse {:.2f} G'.format(peak_memuse))

    if (epoch + 1) % 100 == 0: 
        print('Epoch:{:04d}'.format(epoch + 1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'f1:{:.3f}'.format(acc_val * 100),
            'best f1:{:.3f}'.format(best_val * 100),
            f'Average epoch time: {total_time / (epoch+1-args.start_epoch):.2f}')
            
    if acc_val > best_val:
        best_val = acc_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0        # 原来bad_counter策略是一旦有更好的就清零
    else:
        bad_counter += 1
        if bad_counter == args.patience:
            break


print('parameter number:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
print('best val f1: {:.2f}'.format(best_val * 100))

if best_epoch <= args.label_warmup[0]:
    label_weight = 0.0
elif best_epoch >= args.label_warmup[1]:
    label_weight = 1.0
else:
    label_weight = (best_epoch - args.label_warmup[0]) / (args.label_warmup[1] - args.label_warmup[0])
test_acc = test(label_weight, if_add_label)


print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test f1: {:.2f}".format(test_acc * 100))
