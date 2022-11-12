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
from sklearn.manifold import TSNE
from sklearn import tree
import math
import matplotlib.pyplot as plt


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
def add_labels(graph, idx, label_weight, label_onehot):   # this graph is a subgraph, like "Block(num_src_nodes=131135, num_dst_nodes=130794, num_edges=4034241)"
    feat = graph.srcdata["feat"]
    if label_onehot:
        train_labels = torch.zeros([feat.shape[0], 2 * n_classes], device=device)
    else:
        train_labels = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels[idx] = graph.srcdata["train_labels"][idx] * label_weight
    graph.srcdata["feat"] = torch.cat([feat, train_labels], dim=-1)   # add labels into feature


def cal_sim(experts, feats):
    intra_sim = 0
    inter_sim = 0
    intra_cnt = 0
    inter_cnt = 0
    for i in range(0, len(experts) - 1):
        for j in range(i + 1, len(experts)):
            i_norm = np.linalg.norm(feats[i])
            j_norm = np.linalg.norm(feats[j])
            dot_pro = np.dot(feats[i], feats[j])
            if dot_pro == 0:
                sim = 0
            else:
                sim = dot_pro/(i_norm * j_norm)
            if experts[i] == experts[j]:
                intra_sim += sim
                intra_cnt += 1
            else:
                inter_sim += sim
                inter_cnt += 1
    intra_sim /= intra_cnt
    inter_sim /= inter_cnt
    print("intra sim:{:.3f}, inter sim:{:.3f}, ratio:{:.2f}".format(intra_sim, inter_sim, intra_sim / inter_sim))
    return intra_sim / inter_sim


# 取前一半的点训练，后一半的点预测，看准确率
def decision_tree(experts, feats):
    total_num = len(experts)
    clf = tree.DecisionTreeClassifier(max_depth=15)
    clf.fit(feats[:int(total_num / 2)], experts[:int(total_num / 2)])
    pred = clf.predict(feats[int(total_num / 2):])       # 预测结果
    acc = (pred == experts[int(total_num / 2):]).sum() / len(experts[int(total_num / 2):])
    print('decision tree accuracy:{:.4f}'.format(acc))
    return acc


def analyze_labels(experts, labels, num_expert):
    label_sum = labels.sum(0)      # 每一维度的总label数
    label_rate = label_sum / len(experts)           # 每一维度的总label比例
    expert_label_rate = torch.zeros(num_expert, 112)
    expert_sum = [0] * num_expert
    for i, ex in enumerate(experts):
        expert_label_rate[ex] += labels[i]
        expert_sum[ex] += 1
    for i in range(num_expert):
        expert_label_rate[i] /= expert_sum[i]
        for j in range(112):
            if expert_label_rate[i][j] > 2 * label_rate[j] or expert_label_rate[i][j] < label_rate[j] / 5:
                print(i, j, expert_label_rate[i][j], label_rate[j])


# 换一种度量方式，看一个类内的点是否普遍连成一块，看每个点最近的5个点有多大比例是自己人
def cal_sim2(experts, feats):
    experts_record = [[-1, -1, -1, -1, -1] for _ in range(len(experts))]
    sim_record = [[-1, -1, -1, -1, -1] for _ in range(len(experts))]
    for i in range(0, len(experts) - 1):
        for j in range(i, len(experts)):
            i_norm = np.linalg.norm(feats[i])
            j_norm = np.linalg.norm(feats[j])
            dot_pro = np.dot(feats[i], feats[j])
            if dot_pro == 0:
                sim = 0
            else:
                sim = dot_pro/(i_norm * j_norm)
            i_min = np.min(sim_record[i])
            if sim > i_min:
                i_min_idx = np.argmin(sim_record[i])
                sim_record[i][i_min_idx] = sim
                experts_record[i][i_min_idx] = experts[j]
            j_min = np.min(sim_record[j])
            if sim > j_min:
                j_min_idx = np.argmin(sim_record[j])
                sim_record[j][j_min_idx] = sim
                experts_record[j][j_min_idx] = experts[i]
    
    cnt = 0
    for i, ex in enumerate(experts):
        for j in range(0, 5):
            if experts_record[i][j] == ex:
                cnt += 1
    print("neighbor from same expert number:{}, ratio:{:.4f}".format(cnt, cnt / (len(experts) * 5)))
    return cnt / (len(experts) * 5)


# this function is used to cluster
def cluster(experts_list, moe_emb_list, feature, args):
    tree_accs = []
    sim_ratios = []
    sim_ratios2 = []
    color_list = ['black', 'dimgrey', 'maroon', 'chocolate', 'orange', 'burlywood', 'darkgoldenrod', 'darkkhaki', 
        'olive', 'yellowgreen', 'lightseagreen', 'teal', 'deepskyblue', 'steelblue', 'navy', 'indigo', 'purple']

    plt.figure(figsize=(45, 30))
    
    for layer_n in range(1, 7):
        plt.subplot(2, 3, layer_n)
        plt.title('Layer ' + str(layer_n), size=30)
        print("generating", layer_n)
        feats = moe_emb_list[layer_n - 1]
        experts = experts_list[layer_n - 1]
        feats = feats.cpu()
        idx_list = list(range(0, feats.shape[0]))
        random.shuffle(idx_list)
        experts = [ex.item() for ex in experts]
        experts = np.asarray(experts, dtype=int)

        # acc = decision_tree(experts, feats)     # 决策树分析的时候千万别只取1000个点，会严重过拟合！
        # tree_accs.append(acc)
        if feature == 'label':
            analyze_labels(experts, feats, args.num_expert)

        # idx_list = idx_list[:1000]  # 只取前1000个点算决策树和距离
        # feats = feats[idx_list]
        # experts = experts[idx_list]
        # sim = cal_sim(experts, feats)
        # sim2 = cal_sim2(experts, feats)
        # sim_ratios.append(sim)
        # sim_ratios2.append(sim2)
        # continue
        
        feats = feats[:10000]        # 只取前10000个点算聚类
        experts = experts[:10000]
        if feature == 'output':
            # default: perplexity=30, learning_rate=200
            trans_feats = TSNE(n_components=2, metric="cosine").fit_transform(feats)
        elif feature == 'input':
            trans_feats = TSNE(n_components=2, metric="cosine", perplexity=50, learning_rate=50).fit_transform(feats)
        elif feature == 'origin':
            trans_feats = TSNE(n_components=2, metric="cosine", perplexity=200, learning_rate=500).fit_transform(feats)
        elif feature == 'label':
            trans_feats = TSNE(n_components=2, metric="cosine").fit_transform(feats)

        # plt.figure(figsize=(15, 15))
        feats_draw = []
        for i in range(16):
            feats_draw.append(trans_feats[experts == i])
        for i in range(16):
            plt.scatter(feats_draw[i][:, 0], feats_draw[i][:, 1], label=i, color=color_list[i], alpha=0.6)
        # for j in range(500):      # 标注数据
        #     plt.annotate(experts[j], xy=(trans_feats[j,0], trans_feats[j,1]))
        if layer_n == 6:
            plt.legend(loc='lower right', prop={'size': 24})
        plt.axis('off')

    plt.savefig('visual/{}.png'.format(feature), bbox_inches='tight')
    plt.savefig('visual/{}.pdf'.format(feature), bbox_inches='tight')
    plt.close()
    
    # print("decision tree accuracy:{:.4f}, similarity ratio1:{:.4f}, similarity ratio2:{:.4f}".format(
    #     np.mean(tree_accs), np.mean(sim_ratios), np.mean(sim_ratios2)))


@torch.no_grad()
def evaluate(args, model, dataloader, labels, label_weight):
    model.eval()
    for input_nodes, output_nodes, subgraphs in dataloader:
        subgraphs = [b.to(device) for b in subgraphs]
        new_train_idx= torch.arange(len(output_nodes), len(input_nodes), device=device)
        add_labels(subgraphs[0], new_train_idx, label_weight, args.label_onehot)
        experts_list, moe_in_list, moe_out_list = model.pseudo_forward(subgraphs)
        del model
        if "output" in args.features:
            print('generating information about output')
            cluster(experts_list, moe_out_list, 'output', args)
        if 'input' in args.features:
            print('generating information about input')
            cluster(experts_list, moe_in_list, 'input', args)
        if 'origin' in args.features:
            print('generating information about origin')
            origin_feat_list = [g.dstdata['feat'] for g in subgraphs]
            cluster(experts_list, origin_feat_list, 'origin', args)
        if 'label' in args.features:
            print('generating information about label')
            label_list = [g.dstdata['labels'] for g in subgraphs]
            cluster(experts_list, label_list, 'label', args)
        exit()


def load_model(args, model):
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(str(model), args.lr, args.lr_patience, \
                args.lr_factor, args.wd, args.label_warmup[0], args.label_warmup[1], \
                args.seed, args.match_n_epochs)
    model.load_state_dict(torch.load('parameters/' + model_name + '.pkl'))
    print('load model:', model_name)
    return model


# this function is used for one run (one call for a run)
def run(args, graph, labels, train_idx, val_idx, test_idx):
    model = gen_model(args).to(device)   # acquire model
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_param, 'params')  # original model: 11740552 params
    model = load_model(args, model)

    eval_sampler = MultiLayerNeighborSampler([100 for _ in range(args.n_layers)])
    eval_dataloader = DataLoaderWrapper(NodeDataLoader(
        graph.cpu(), torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]), eval_sampler,
        batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=65536),  # 为老版本dgl
        num_workers=0    # 如果10的时候报错就改成0
    ))
    
    label_weight = 0.0
    epoch = args.match_n_epochs
    if epoch >= args.label_warmup[1]:
        label_weight = 1.0
    elif epoch >= args.label_warmup[0]:
        label_weight = (epoch - args.label_warmup[0]) / (args.label_warmup[1] - args.label_warmup[0])
    evaluate(args, model, eval_dataloader, labels, label_weight)


def main():
    global device
    argparser = argparse.ArgumentParser(
       "GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--n_batchs", type=int, default=10)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
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
    argparser.add_argument("--num_expert", type=int, default=8, help="number of experts")
    argparser.add_argument("--top_k", type=int, default=2, help="selected number of experts")
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
    argparser.add_argument("--expert_drop", type=float, default=0.0)
    argparser.add_argument("--gat2", action="store_true")
    argparser.add_argument("--features", type=str, nargs='+', default=[])
    argparser.add_argument("--lr_patience", type=int, default=50, help="lr patience for scheduler")
    argparser.add_argument("--lr_factor", type=float, default=0.75, help="lr factor for schedulter")
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
    print(graph)
    graph = preprocess(graph, labels, train_idx, args.label_onehot)
    # count_labels(labels)
    graph, labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx))  # move them to gpu
    run(args, graph, labels, train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()
