import random
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# set random seeds for everything
def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)



def plot(args, train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses):
    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, args.n_epochs, 100))
    ax.set_yticks(np.linspace(0, 1.0, 101))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip([train_scores, val_scores, test_scores], ["train score", "val score", "test score"]):
        plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_score_{n_running}.png")

    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, args.n_epochs, 100))
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip(
        [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
    ):
        plt.plot(range(1, args.n_epochs + 1, args.log_every), y, label=label, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"gat_loss_{n_running}.png")


# 这个类暂时没用
class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


# this wrapper helps to repeattedly iterate the DataLoader
class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:   # if stop iteration, then generate a new iterator
            self.iter = iter(self.dataloader)
            raise StopIteration() from None


class BatchSampler(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            shuf = torch.randperm(self.n).split(self.batch_size)  # torch.randperm(n)返回[0,n-1]的随机排列
            # split为将[0,n-1]的随机排列从前向后切成一堆份，每份是batch_size大小
            for shuf_batch in shuf:
                yield shuf_batch
            yield None
