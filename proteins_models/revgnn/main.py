import __init__
import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
# from model import DeeperGCN
# from model_geq import DEQGCN
from model_rev import RevGCN
# from model_revwt import WTRevGCN
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes
import logging
from torch.utils.tensorboard import SummaryWriter


def train(data, dataset, model, optimizer, criterion, device, epoch=-1):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr, epoch=epoch)

        target = train_y[inter_idx].to(device)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, device, epoch=-1):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(x, sg_nodes_idx, sg_edges[idx].to(device),
                         sg_edges_attr, epoch=epoch).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result


def main():
    args = ArgsInit().save_exp()
    logging.getLogger().setLevel(logging.INFO)
    writer = SummaryWriter(log_dir=args.save)

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info('%s' % device)
    
    dataset = OGBNDataset(dataset_name=args.dataset, root='../gat_bot_ngnn/dataset')
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    logging.info('%s' % args)

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []

    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                               cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts,
                                                 cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    sub_dir = 'random-train_{}-test_{}-num_evals_{}'.format(args.cluster_number,
                                                            args.valid_cluster_number,
                                                            args.num_evals)
    logging.info(sub_dir)

    if args.backbone == 'deepergcn':
        # model = DeeperGCN(args).to(device)
        pass
    # elif args.backbone == 'deq':
        # model = DEQGCN(args).to(device)
    # elif args.backbone == 'revwt':
        # model = WTRevGCN(args).to(device)
    elif args.backbone == 'rev':
        model = RevGCN(args).to(device)
    else:
        raise Exception("unkown backbone")

    logging.info('# of params: {}'.format(sum(p.numel() for p in model.parameters())))
    logging.info('# of learnable params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()
    train_time = 0

    for epoch in range(1, args.epochs + 1):
        # do random partition every epoch
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                     cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)
        tic = time.time()
        epoch_loss = train(data, dataset, model, optimizer, criterion, device, epoch=epoch)
        toc = time.time()
        train_time += toc - tic
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        if epoch < 10:
            peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
            logging.info('Peak memuse {:.2f} G'.format(peak_memuse))
        torch.cuda.empty_cache()

        model.print_params(epoch=epoch)

        with torch.cuda.amp.autocast():
            result = multi_evaluate(valid_data_list, dataset, model,
                                    evaluator, device, epoch=epoch)

        if epoch % 5 == 0:
            logging.info('%s' % result)
            print(f"Epoch: {epoch}/{args.epochs}, Average epoch time: {train_time / epoch:.2f}s")

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']
        writer.add_scalar('stats/train_rocauc', train_result, epoch)
        writer.add_scalar('stats/valid_rocauc', valid_result, epoch)
        writer.add_scalar('stats/test_rocauc', test_result, epoch)

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            save_ckpt(model, optimizer, round(epoch_loss, 4),
                      epoch,
                      args.model_save_path, sub_dir,
                      name_post='valid_best')

        if train_result > results['highest_train']:
            results['highest_train'] = train_result

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%d-%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
