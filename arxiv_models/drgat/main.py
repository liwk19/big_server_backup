import argparse
import math
import os
import random
import time
import logging
import uuid
import sys
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import loss_kd_only
from model import DRGAT


epsilon = 1 - math.log(2)
device = None
dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset, args):
    global n_node_feats, n_classes
    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # Replace node features here
    if args.pretrain_path != 'None':
        graph.ndata["feat"] = torch.tensor(np.load(args.pretrain_path)).float()
        print("Pretrained node feature loaded! Path: {}".format(args.pretrain_path))
    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats
    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    graph.create_formats_()
    return graph


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.backbone == 'drgat':
        model = DRGAT(
            n_node_feats_, n_classes, n_hidden=args.n_hidden, n_layers=args.n_layers,
            n_heads=args.n_heads, activation=F.relu, dropout=args.dropout, hid_drop=args.hid_drop,
            input_drop=args.input_drop, attn_drop=args.attn_drop, edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst, use_symmetric_norm=args.use_norm,
            num_expert=args.num_expert, top_k=args.top_k, moe_widget=args.moe_widget,
            expert_drop=args.expert_drop, gate=args.gate
        )

    else:
        raise Exception("Unknown backnone")
    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:      # 初始有个warmup
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator, 
    mode='teacher', teacher_output=None):
    model.train()
    if mode == 'student':
        assert teacher_output != None

    alpha = args.alpha
    temp = args.temp
    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_labels_idx = train_idx[mask]    # mask_rate是把train_label加入feature的概率
        train_pred_idx = train_idx[~mask]   # 没加入feature的就是要预测的
        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    if args.n_label_iters > 0:
        with torch.no_grad():
            pred, gate_std = model(graph, feat)
    else:
        pred, gate_std = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            # torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, gate_std = model(graph, feat)

    # 只取最后一次gate_std算进loss，不要每个都放进去算，没有必要
    if mode == 'teacher':
        loss = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
        if args.moe_alpha > 0:
            loss += args.moe_alpha * gate_std
    elif mode == 'student':
        loss_gt = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
        loss_kd = loss_kd_only(pred, teacher_output, temp)
        loss = loss_gt * (1 - alpha) + loss_kd * alpha
        if args.moe_alpha > 0:
            loss += args.moe_alpha * gate_std
    else:
        raise Exception('unkown mode')

    loss.backward()
    optimizer.step()
    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()
    feat = graph.ndata["feat"]
    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)
    pred, gate_std = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, gate_std = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])
    
    return (
        evaluator(pred[train_idx], labels[train_idx]), evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss, val_loss, test_loss, pred
    )


def save_pred(pred, run_num, kd_dir):
    if not os.path.exists(kd_dir):
        os.makedirs(kd_dir)
    fname = os.path.join(kd_dir, 'best_pred_run{}.pt'.format(run_num))
    torch.save(pred.cpu(), fname)


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]
    # kd mode
    mode = args.mode
    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss, best_epoch = 0, 0, float("inf"), 0
    final_pred = None
    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    val600, val800, val1000, val1200, val1400, val1600, val1800 = 0, 0, 0, 0, 0, 0, 0
    test600, test800, test1000, test1200, test1400, test1600, test1800 = 0, 0, 0, 0, 0, 0, 0

    if mode == 'student':
        teacher_output = torch.load('./{}/best_pred_run{}.pt'.format(args.kd_dir, n_running + args.seed)).cpu().cuda()
        print('./{}/best_pred_run{}.pt'.format(args.kd_dir, n_running + args.seed))
        teacher_acc = evaluator_wrapper(teacher_output[test_idx], labels[test_idx])
        print('teacher test acc:', teacher_acc)
    else:
        teacher_output = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        adjust_learning_rate(optimizer, args.lr, epoch)
        acc, loss = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, 
            evaluator_wrapper, mode=mode, teacher_output=teacher_output)
        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )
        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:     # 用val_loss去判断最佳epoch，是否改成val_acc更合理？
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            best_epoch = epoch
            if mode == 'teacher':
                save_pred(final_pred, n_running + args.seed, args.kd_dir)

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            logging.info(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc/Best epoch: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}/{best_epoch}\n"
            )
            peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
            logging.info('Peak memuse {:.2f} G'.format(peak_memuse))

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)
        
        if epoch == 600:
            val600 = best_val_acc
            test600 = final_test_acc
        if epoch == 800:
            val800 = best_val_acc
            test800 = final_test_acc
        if epoch == 1000:
            val1000 = best_val_acc
            test1000 = final_test_acc
        elif epoch == 1200:
            val1200 = best_val_acc
            test1200 = final_test_acc
        elif epoch == 1400:
            val1400 = best_val_acc
            test1400 = final_test_acc
        elif epoch == 1600:
            val1600 = best_val_acc
            test1600 = final_test_acc
        elif epoch == 1800:
            val1800 = best_val_acc
            test1800 = final_test_acc

    logging.info("*" * 50)
    logging.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    logging.info("*" * 50)

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running + args.seed}.pt")
    return val600, val800, val1000, val1200, val1400, val1600, val1800, best_val_acc, test600, test800, test1000, test1200, test1400, test1600, test1800, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon
    argparser = argparse.ArgumentParser("DRGAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=1, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--hid-drop", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=50, help="log every LOG_EVERY epochs")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--save", type=str, default='exp', help="save exp")
    argparser.add_argument('--backbone', type=str, default='drgat', help='gcn backbone [deepergcn, drgcn, drgat]')
    argparser.add_argument("--kd_dir", type=str, default='./kd', help="kd path for pred")
    argparser.add_argument("--mode", type=str, default='teacher', help="kd mode [teacher, student]", choices=['teacher', 'student'])
    argparser.add_argument("--alpha", type=float, default=0.5, help="ratio of kd loss")
    argparser.add_argument("--temp", type=float, default=1.0, help="temperature of kd")
    argparser.add_argument('--data_root_dir', type=str, default='default', help="dir_path for saving graph data.")
    argparser.add_argument("--pretrain_path", type=str, default='None', help="path for pretrained node features")
    argparser.add_argument('--num_expert', type=int, default=1)
    argparser.add_argument('--top_k', type=int, default=1)
    argparser.add_argument('--no_teacher_moe', action='store_true', default=False)
    argparser.add_argument("--moe_widget", type=str, nargs='+', default=[], 
        choices=['fc', 'attn_l', 'ngnn'])
    argparser.add_argument('--expert_drop', type=float, default=0.0)
    argparser.add_argument("--gate", type=str, default='naive', help="FMOE's gate", 
        choices=['naive', 'mlp', 'gcn', 'sage', 'gat'])
    argparser.add_argument("--moe_alpha", type=float, default=0.0)
    args = argparser.parse_args()
    print(args)

    # Adjust kd_dir here
    ori_kd_dir = args.kd_dir
    args.kd_dir = '{}/L{}-H{}-moe_{}-ngnn_{}'.format(args.kd_dir, args.n_layers, args.n_hidden, args.num_expert > 1, args.moe_widget==['ngnn'])
    args.save = '{}/L{}-H{}-moe_{}-ngnn_{}'.format(args.kd_dir, args.n_layers, args.n_hidden, args.num_expert > 1, args.moe_widget==['ngnn'])
    args.save = 'log/{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if args.mode == 'student':
        args.kd_dir = '{}/L{}-H{}-moe_{}-ngnn_{}'.format(ori_kd_dir, args.n_layers, args.n_hidden, not args.no_teacher_moe, args.moe_widget==['ngnn'])

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, args)
    graph = preprocess(graph)
    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )
    logging.info(args)
    logging.info(f"Number of params: {count_parameters(args)}")

    # run
    val_accs_600, val_accs_800, val_accs_1000, val_accs_1200, val_accs_1400, val_accs_1600, val_accs_1800, val_accs, \
        test_accs_600, test_accs_800, test_accs_1000, test_accs_1200, test_accs_1400, test_accs_1600, test_accs_1800, test_accs = [], \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(args.n_runs):
        seed(args.seed + i)
        val600, val800, val1000, val1200, val1400, val1600, val1800, val_acc, \
            test600, test800, test1000, test1200, test1400, test1600, test1800, test_acc = run(args, \
            graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_accs_600.append(val600)
        val_accs_800.append(val800)
        val_accs_1000.append(val1000)
        val_accs_1200.append(val1200)
        val_accs_1400.append(val1400)
        val_accs_1600.append(val1600)
        val_accs_1800.append(val1800)
        val_accs.append(val_acc)
        test_accs_600.append(test600)
        test_accs_800.append(test800)
        test_accs_1000.append(test1000)
        test_accs_1200.append(test1200)
        test_accs_1400.append(test1400)
        test_accs_1600.append(test1600)
        test_accs_1800.append(test1800)
        test_accs.append(test_acc)

        logging.info(f"Average val accuracy: {100 * np.mean(val_accs):.2f} ± {100 * np.std(val_accs):.2f}")
        logging.info(f"Average test accuracy 600: {100 * np.mean(test_accs_600):.2f} ± {100 * np.std(test_accs_600):.2f}")
        logging.info(f"Average test accuracy 800: {100 * np.mean(test_accs_800):.2f} ± {100 * np.std(test_accs_800):.2f}")
        logging.info(f"Average test accuracy 1000: {100 * np.mean(test_accs_1000):.2f} ± {100 * np.std(test_accs_1000):.2f}")
        logging.info(f"Average test accuracy 1200: {100 * np.mean(test_accs_1200):.2f} ± {100 * np.std(test_accs_1200):.2f}")
        logging.info(f"Average test accuracy 1400: {100 * np.mean(test_accs_1400):.2f} ± {100 * np.std(test_accs_1400):.2f}")
        logging.info(f"Average test accuracy 1600: {100 * np.mean(test_accs_1600):.2f} ± {100 * np.std(test_accs_1600):.2f}")
        logging.info(f"Average test accuracy 1800: {100 * np.mean(test_accs_1800):.2f} ± {100 * np.std(test_accs_1800):.2f}")
        logging.info(f"Average test accuracy: {100 * np.mean(test_accs):.2f} ± {100 * np.std(test_accs):.2f}")

    logging.info(args)
    logging.info(f"Runned {args.n_runs} times")
    logging.info("Val Accs 800:")
    logging.info(val_accs_800)
    logging.info("Val Accs 1000:")
    logging.info(val_accs_1000)
    logging.info("Val Accs 1200:")
    logging.info(val_accs_1200)
    logging.info("Val Accs 1400:")
    logging.info(val_accs_1400)
    logging.info("Val Accs 1600:")
    logging.info(val_accs_1600)
    logging.info("Val Accs 1800:")
    logging.info(val_accs_1800)
    logging.info("Val Accs:")
    logging.info(val_accs)
    logging.info("Test Accs 800:")
    logging.info(test_accs_800)
    logging.info("Test Accs 1000:")
    logging.info(test_accs_1000)
    logging.info("Test Accs 1200:")
    logging.info(test_accs_1200)
    logging.info("Test Accs 1400:")
    logging.info(test_accs_1400)
    logging.info("Test Accs 1600:")
    logging.info(test_accs_1600)
    logging.info("Test Accs 1800:")
    logging.info(test_accs_1800)
    logging.info("Test Accs:")
    logging.info(test_accs)
    print()
    logging.info(f"Average val accuracy: {100 * np.mean(val_accs):.2f} ± {100 * np.std(val_accs):.2f}")
    logging.info(f"Average test accuracy 600: {100 * np.mean(test_accs_600):.2f} ± {100 * np.std(test_accs_600):.2f}")
    logging.info(f"Average test accuracy 800: {100 * np.mean(test_accs_800):.2f} ± {100 * np.std(test_accs_800):.2f}")
    logging.info(f"Average test accuracy 1000: {100 * np.mean(test_accs_1000):.2f} ± {100 * np.std(test_accs_1000):.2f}")
    logging.info(f"Average test accuracy 1200: {100 * np.mean(test_accs_1200):.2f} ± {100 * np.std(test_accs_1200):.2f}")
    logging.info(f"Average test accuracy 1400: {100 * np.mean(test_accs_1400):.2f} ± {100 * np.std(test_accs_1400):.2f}")
    logging.info(f"Average test accuracy 1600: {100 * np.mean(test_accs_1600):.2f} ± {100 * np.std(test_accs_1600):.2f}")
    logging.info(f"Average test accuracy 1800: {100 * np.mean(test_accs_1800):.2f} ± {100 * np.std(test_accs_1800):.2f}")
    logging.info(f"Average test accuracy: {100 * np.mean(test_accs):.2f} ± {100 * np.std(test_accs):.2f}")
    logging.info(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
