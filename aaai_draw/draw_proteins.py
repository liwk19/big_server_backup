import numpy as np
import matplotlib.pyplot as plt

single_ngnn = [87.50, 87.91, 88.04, 88.00]
single_moe = [87.90, 88.04, 88.28, 88.26]
double_ngnn = [87.96, 88.36, 88.57, 88.64]
double_moe = [88.22, 88.55, 88.68, 88.71]
x = [60, 80, 100, 120]

single_ngnn_param = np.asarray([2203612, 3878432, 6023652, 8639272])
single_moe_param = np.asarray([13969228, 24749888, 38592948, 55498408])
double_ngnn_param = np.asarray([4540732, 8031392, 12510852, 17979112])
double_moe_param = np.asarray([51363148, 91197248, 142388148, 204935848])

plt.figure(figsize=(12, 6))
# plt.xlabel('Hidden size', size=20)
plt.ylabel('Test ROC-AUC score', size=20)

plt.scatter(x, single_ngnn, label='single-layer NGNN', color='skyblue', linewidth=3)
plt.plot(x, single_ngnn, color='skyblue')
plt.scatter(x, single_moe, label='single-layer MoE', color='blue', linewidth=3)
plt.plot(x, single_moe, color='blue')
plt.scatter(x, double_ngnn, label='double-layer NGNN', color='salmon', linewidth=3)
plt.plot(x, double_ngnn, color='salmon')
plt.scatter(x, double_moe, label='double-layer MoE', color='red', linewidth=3)
plt.plot(x, double_moe, color='red')
plt.xticks(x, size=15)
plt.yticks(size=15)
plt.ylim((87.4, 88.8))
plt.grid(linestyle=':', axis='y')
plt.legend(loc='lower right', prop={'size': 15})
plt.savefig('proteins.pdf')
plt.savefig('proteins.png')
