#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader       # 使用dgl 0.7.2
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn

from models import GAT
from utils import BatchSampler, DataLoaderWrapper, seed, plot
import wandb

device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 8, 8, 112


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


# load data and evaluator
def load_data(dataset):
    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels   # set labels
    return graph, labels, train_idx, val_idx, test_idx, evaluator


# preprocessthe graph to add some features
def preprocess(graph, labels, train_idx, label_onehot=False):
    global n_node_feats
    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
    n_node_feats = graph.ndata["feat"].shape[-1]

    # Only the labels in the training set are used as features, while others are filled with zeros.
    if label_onehot:
        graph.ndata["train_labels"] = torch.zeros([graph.number_of_nodes(), 2 * n_classes], dtype=int)
        for i in range(112):
            graph.ndata["train_labels"][train_idx, 2 * i + labels[train_idx, i]] = 1
    else:
        graph.ndata["train_labels"] = torch.zeros([graph.number_of_nodes(), n_classes], dtype=int)
        graph.ndata["train_labels"][train_idx] = labels[train_idx]
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)
    graph.create_formats_()
    print('preprocess')
    print(graph)
    return graph


# generate the GAT model from args
def gen_model(args):
    if args.n_epochs < args.label_warmup[0]:  # 不用label
        n_node_feats_ = n_node_feats
    else:
        if args.label_onehot:
            n_node_feats_ = n_node_feats + 2 * n_classes
        else:
            n_node_feats_ = n_node_feats + n_classes       # MyAlter
    model = GAT(n_node_feats_, n_edge_feats, n_classes, n_layers=args.n_layers, n_heads=args.n_heads,
        n_hidden=args.n_hidden, edge_emb=16, activation=F.relu, dropout=args.dropout,
        input_drop=args.input_drop, attn_drop=args.attn_drop, edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst, num_expert=args.num_expert, top_k=args.top_k,
        fmoe2=args.fmoe2, pred_fmoe=args.pred_fmoe, 
        moe_drp=args.moe_drp, gate=args.gate, moe_widget=args.moe_widget, fmoe22=args.fmoe2_src_dst_fc,
        expert_drop=args.expert_drop, gat2=args.gat2
    )
    return model


# add labels into node features
def add_labels(graph, idx, label_weight, label_onehot, args):   # this graph is a subgraph, like "Block(num_src_nodes=131135, num_dst_nodes=130794, num_edges=4034241)"
    if args.n_epochs < args.label_warmup[0]:
        return
    feat = graph.srcdata["feat"]
    if label_onehot:
        train_labels = torch.zeros([feat.shape[0], 2 * n_classes], device=device)
    else:
        train_labels = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels[idx] = graph.srcdata["train_labels"][idx] * label_weight
    graph.srcdata["feat"] = torch.cat([feat, train_labels], dim=-1)   # add labels into feature


# train the model for one epoch
def train(args, model, dataloader, _labels, _train_idx, criterion, optimizer, _evaluator, label_weight):
    model.train()
    loss_sum, loss1_sum, loss2_sum, loss3_sum, total = 0, 0, 0, 0, 0

    for input_nodes, output_nodes, subgraphs in dataloader:   # input_nodes is an array of about 131100(not fixed) nodes, output_nodes is an array of 8662/8661 members
        # len(input_nodes) equals to subgraphs[0]的点数, input_nodes里为原图里被sample出的点的编号
        # output_nodes为0-86618（train set）里的点的index，就是把原train set分成10等份
        # subgraphs = [b.to(device) for b in subgraphs]   # 6 subgraphs, the first of which has input_nodes number of nodes
        # subgraphs最后一层就是8619，前面每向前一层就是扩大一层邻居。第一层的输入为6-hop邻居数量，第一层的边是从6-hop邻居指向5-hop邻居
        # 最后一层的边是从1-hop邻居指向哪些输出点自身。subgraphs数量=层数。
        # 每一个subgraph有number_of_dst_nodes()为(n+1)-hop邻居，有number_of_src_nodes()为n-
        train_pred_idx = torch.arange(len(output_nodes), device=device)   # torch.arrange() is similar to normal range(), [0, 8661]
        train_labels_idx = torch.arange(len(output_nodes), len(input_nodes), device=device)
        add_labels(subgraphs[0], train_labels_idx, label_weight, args.label_onehot, args)
        
        pred, gate_std_att, gate_std_ngnn = model(subgraphs)
        loss1 = criterion(pred[train_pred_idx], subgraphs[-1].dstdata["labels"][train_pred_idx].float())
        loss2 = gate_std_att
        loss3 = gate_std_ngnn
        if args.attn_alpha == 0 and args.ngnn_alpha == 0:
            loss = loss1
        elif args.attn_alpha == 0 and args.ngnn_alpha != 0:
            loss = loss1 + args.ngnn_alpha * loss3
        elif args.attn_alpha != 0 and args.ngnn_alpha == 0:
            loss = loss1 + args.attn_alpha * loss2
        elif args.attn_alpha != 0 and args.ngnn_alpha != 0:
            loss = loss1 + args.attn_alpha * loss2 + args.ngnn_alpha * loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = len(train_pred_idx)
        loss_sum += loss.item() * count
        loss1_sum += loss1.item() * count
        loss2_sum += loss2.item() * count
        loss3_sum += loss3.item() * count
        total += count
        torch.cuda.empty_cache()
    
    return loss_sum / total, loss1_sum / total, loss2_sum / total, loss3_sum / total   # weighted sum of different epochs


@torch.no_grad()
def evaluate(args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator, label_weight):
    model.eval()
    preds = torch.zeros(labels.shape).to(device)
    # Due to the memory capacity constraints, we use sampling for inference and calculate the average of the predictions 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        for input_nodes, output_nodes, subgraphs in dataloader:   # 共3组，因为1组为65536个output_nodes
            # len(input_nodes)为132000左右，len(output_nodes) = 65536或最后余数
            # subgraphs = [b.to(device) for b in subgraphs]
            # new_train_idx = list(range(len(input_nodes)))  # MyAlter：发现之前evaluate不太对？
            new_train_idx= torch.arange(len(output_nodes), len(input_nodes), device=device)
            add_labels(subgraphs[0], new_train_idx, label_weight, args.label_onehot, args)
            pred, gate_std_att, gate_std_ngnn = model(subgraphs)
            preds[output_nodes] += pred  # output_nodes被分成10等份，正好10次下来就把preds填满了
            torch.cuda.empty_cache()

    preds /= eval_times
    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()
    # binary_preds = preds.sigmoid().round()   # 从用的loss可以看出，loss里是先过了一个sigmoid才输出的。
    # 所以我这里也先sigmoid，然后用round四舍五入
    # acc = (binary_preds == labels)

    return (
        evaluator(preds[train_idx], labels[train_idx]),  # train rocauc score
        evaluator(preds[val_idx], labels[val_idx]),   # val rocauc score
        evaluator(preds[test_idx], labels[test_idx]),   # test rocauc score
        train_loss, val_loss, test_loss, preds
    )


def load_model(args, model, n_running):
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(str(model), args.lr, args.lr_patience, \
                args.lr_factor, args.wd, args.label_warmup[0], args.label_warmup[1], \
                n_running + args.seed, args.match_n_epochs)
    print('load model:', model_name)
    model.load_state_dict(torch.load('parameters/' + model_name + '.pkl'))
    save_info = np.load('logs/' + model_name + '.npy')
    return model, save_info


# this function is used for a run (one call for a run). labels: [132534, 112]; train_idx: [86619]; val_idx: [21236]; test_idx: [24679]
def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    model = gen_model(args).to(device)   # acquire model
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_param:', num_param)

    if args.match_n_epochs > 0:
        model, save_info = load_model(args, model, n_running)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # lr_scheduler可以动态调整lr，一般是越往后lr降低。mode为max表示score不再增大时降低学习率，patience为连续50轮不在增大则降，factor为乘的权重，verbose为触发条件后是否print
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    if args.match_n_epochs > 0:
        val_scores = save_info[2]
        for i in range(args.eval_every - 1):       # 模拟最初几次还没eval时，val_score是0
            lr_scheduler.step(0)
        for v in val_scores[:-1]:          # 还原之前val_score给lr_scheduler造成的影响
            for i in range(args.eval_every):
                lr_scheduler.step(v)
        lr_scheduler.step(val_scores[-1])

    train_batch_size = (len(train_idx) + 9) // args.n_batchs   # train_idx have 86619 members, so train_batch_size = 8662
    train_sampler = MultiLayerNeighborSampler([32 for _ in range(args.n_layers)])
    eval_sampler = MultiLayerNeighborSampler([100 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    # evaluator_wrapper: send in y_pred and y_true, return rocauc score
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]
    criterion = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoaderWrapper(NodeDataLoader(
        graph, train_idx, train_sampler,
        batch_sampler=BatchSampler(len(train_idx), batch_size=train_batch_size),  # 为老版本dgl（0.7.2）
        num_workers=0    # 如果10的时候报错就改成0
    ))
    eval_dataloader = DataLoaderWrapper(NodeDataLoader(
        graph, torch.cat([train_idx, val_idx, test_idx]), eval_sampler,
        batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=65536),  # 为老版本dgl
        num_workers=0    # 如果10的时候报错就改成0
    ))

    # train_dataloader = DataLoaderWrapper(NodeDataLoader(
    #     graph.cpu(), train_idx.cpu(), train_sampler,
    #     batch_sampler=BatchSampler(len(train_idx), batch_size=train_batch_size),  # 为老版本dgl（0.7.2）
    #     num_workers=0    # 如果10的时候报错就改成0
    # ))
    # eval_dataloader = DataLoaderWrapper(NodeDataLoader(
    #     graph.cpu(), torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]), eval_sampler,
    #     batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=65536),  # 为老版本dgl
    #     num_workers=0    # 如果10的时候报错就改成0
    # ))

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0
    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None

    if args.match_n_epochs > 0:
        losses = list(save_info[0])
        train_scores = list(save_info[1])
        val_scores = list(save_info[2])
        test_scores = list(save_info[3])
        val_score = val_scores[-1]
        best_val_score = np.max(val_scores)
        final_test_score = test_scores[np.argmax(val_scores)]
        print('load model, val_score:{}, best_val_score:{}, final_test_score:{}'.format(val_score, 
            best_val_score, final_test_score))

    for epoch in range(args.match_n_epochs + 1, args.n_epochs + 1):
        tic = time.time()
        label_weight = 0.0
        if epoch >= args.label_warmup[1]:
            label_weight = 1.0
        elif epoch >= args.label_warmup[0]:
            label_weight = (epoch - args.label_warmup[0]) / (args.label_warmup[1] - args.label_warmup[0])
        loss, loss1, loss2, loss3 = train(args, model, train_dataloader, labels, train_idx, criterion, optimizer, evaluator_wrapper, label_weight)
        toc = time.time()
        total_time += toc - tic

        if epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper, label_weight
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                final_pred = pred
            
            wandb.log({'train_score': train_score,
                'val_score': val_score,
                'test_score': test_score,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'best_val_score': best_val_score,
                'final_test_score': final_test_score})

            if epoch % args.log_every == 0:   # print information
                print(
                    f"Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / (epoch - args.match_n_epochs):.2f}s, "
                    f"Loss: {loss:.4f}/ Loss1: {loss1:.4f}/ Loss2: {loss2:.4f}/ Loss3: {loss3:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}\n"
                )

            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)
        
        if epoch < 10:
            peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
            print('Peak memuse {:.2f} G'.format(peak_memuse))

        if epoch % 200 == 0:
            model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(str(model), args.lr, args.lr_patience, \
                args.lr_factor, args.wd, args.label_warmup[0], args.label_warmup[1], \
                n_running + args.seed, epoch)
            torch.save(model.state_dict(), 'parameters/{}.pkl'.format(model_name))
            save_info = [losses, train_scores, val_scores, test_scores]
            save_info = np.asarray(save_info)
            np.save("logs/" + model_name + ".npy", save_info)
            print('save model:', model_name)
        
        lr_scheduler.step(val_score)
    
    print("*" * 50)
    print(f"Best val score: {best_val_score}, Final test score: {final_test_score}")
    print("*" * 50)
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(str(model), args.lr, args.lr_patience, \
                args.lr_factor, args.wd, args.label_warmup[0], args.label_warmup[1], \
                n_running + args.seed, args.n_epochs)
    print('save model:', model_name)
    torch.save(model.state_dict(), 'parameters/{}.pkl'.format(model_name))
    
    save_info = [losses, train_scores, val_scores, test_scores]
    save_info = np.asarray(save_info)
    np.save("logs/" + model_name + ".npy", save_info)
    print('current lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    print('parameter number:', num_param)

    if args.plot:
        plot(args, train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses)
    os.makedirs("./output", exist_ok=True)
    torch.save(F.softmax(final_pred, dim=1), './output/{}.pt'.format(model_name))

    wandb.finish()
    return best_val_score, final_test_score


def count_labels(labels):
    a = labels.sum(0)   # 得到112个label分别的数量
    a = labels.sum(1)   # 得到每个点的为1的label数量，之后b是统计这个数据的分布
    b = [0]*113
    for i in a:
        b[i] += 1
    print(b)


def main():
    global device
    argparser = argparse.ArgumentParser(
       "GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--n_batchs", type=int, default=10)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--match_n_epochs", type=int, default=0, help="number of epochs to match")
    argparser.add_argument("--label_warmup", nargs='+', type=int, default=[200, 800])
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=6, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.008, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--num_expert", type=int, default=8, help="number of experts")
    argparser.add_argument("--top_k", type=int, default=2, help="selected number of experts")
    argparser.add_argument("--lr_patience", type=int, default=50, help="lr patience for scheduler")
    argparser.add_argument("--lr_factor", type=float, default=0.75, help="lr factor for schedulter")
    argparser.add_argument("--fmoe2", type=int, default=0, help="FMoELinear's hidden layer's size, 0 means no hidden layer")
    argparser.add_argument("--pred_fmoe", action="store_true", help="whether to use FMoELinear as classifier head")
    argparser.add_argument("--moe_drp", type=float, default=0, help="FMoEMLP's dropout")
    argparser.add_argument("--gate", type=str, default='naive', help="FMOE's gate", 
        choices=['naive', 'noisy', 'swipe', 'switch', 'mlp', 'gcn', 'sage', 'sage2', 'gat', 'gat2'])
    argparser.add_argument("--moe_widget", type=str, nargs='+', default=[], 
        choices=['ngnn', 'src_fc', 'dst_fc', 'attn_src_fc', 'attn_dst_fc', 'attn_edge_fc', 'encoder'])
    argparser.add_argument("--label_onehot", action="store_true")
    argparser.add_argument("--fmoe2_src_dst_fc", type=int, default=0, help="src_fc and dst_fc's moe hidden layer size")
    argparser.add_argument("--extra_feature", type=str, nargs='+', default=[])
    argparser.add_argument("--attn_alpha", type=float, default=0.0)
    argparser.add_argument("--ngnn_alpha", type=float, default=0.0)
    argparser.add_argument("--expert_drop", type=float, default=0.0)
    argparser.add_argument("--gat2", action="store_true")
    args = argparser.parse_args()
    print(args)

    assert len(args.label_warmup) == 2, 'label_warmup must have 2 numbers'
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print('device', device)

    # load data & preprocess
    print("Loading data")
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph, labels, train_idx, args.label_onehot)
    # count_labels(labels)
    graph, labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx))  # move them to gpu

    # run
    wandb.init(project="GAT_BOT_NGNN_proteins", config=vars(args))   # vars把Namespace对象转字典
    val_scores, test_scores = [], []

    for i in range(args.n_runs):
        print("Running", i)
        seed(args.seed + i)   # set different seeds for different runs
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")


if __name__ == "__main__":
    main()
