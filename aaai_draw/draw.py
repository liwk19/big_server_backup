import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
single_ngnn = [87.50, 87.91, 88.04, 88.00]
single_moe = [87.90, 88.04, 88.28, 88.26]
double_ngnn = [87.96, 88.36, 88.57, 88.64]
double_moe = [88.22, 88.55, 88.68, 88.71]
x = [60, 80, 100, 120]

single_ngnn_param = np.asarray([2203612, 3878432, 6023652, 8639272])
single_moe_param = np.asarray([13969228, 24749888, 38592948, 55498408])
double_ngnn_param = np.asarray([4540732, 8031392, 12510852, 17979112])
double_moe_param = np.asarray([51363148, 91197248, 142388148, 204935848])

# plt.xlabel('Hidden size', size=20)
plt.ylabel('Test ROC-AUC (%)', size=20)
plt.title('ogbn-proteins', size=22)

plt.scatter(x, single_ngnn, label='single-layer NGNN', color='skyblue', linewidth=5)
plt.plot(x, single_ngnn, color='skyblue', linewidth=2)
plt.scatter(x, single_moe, label='single-layer GMoE', color='blue', linewidth=5)
plt.plot(x, single_moe, color='blue', linewidth=2)
plt.scatter(x, double_ngnn, label='double-layer NGNN', color='lightsalmon', linewidth=5)
plt.plot(x, double_ngnn, color='lightsalmon', linewidth=2)
plt.scatter(x, double_moe, label='double-layer GMoE', color='red', linewidth=5)
plt.plot(x, double_moe, color='red', linewidth=2)
plt.xticks(x, size=16)
plt.yticks(size=16)
plt.ylim((87.4, 88.8))
plt.grid(linestyle=':', axis='y')
plt.legend(loc='lower right', prop={'size': 16})




plt.subplot(2, 1, 2)
origin = [54.47, 56.25, 57.19]
moe = [54.78, 56.47, 57.39]
x = [1,2,3]
x_ori = [128, 256, 512]
origin_no = [53.98, 55.85, 56.71]     # no ComplEx feature
moe_no = [54.20, 55.99, 56.82]

plt.xlabel('Hidden size', size=20)
plt.ylabel('Test accuracy (%)', size=20)
plt.title('ogbn-mag', size=22)

plt.scatter(x, origin, label='SeHGNN+ComplEx', color='skyblue', linewidth=5)
plt.plot(x, origin, color='skyblue', linewidth=2)
plt.scatter(x, moe, label='SeHGNN+ComplEx+GMoE', color='blue', linewidth=5)
plt.plot(x, moe, color='blue', linewidth=2)
plt.scatter(x, origin_no, label='SeHGNN', color='lightsalmon', linewidth=5)
plt.plot(x, origin_no, color='lightsalmon', linewidth=2)
plt.scatter(x, moe_no, label='SeHGNN+GMoE', color='red', linewidth=5)
plt.plot(x, moe_no, color='red', linewidth=2)
plt.xticks(x, x_ori, size=16)
plt.yticks(size=16)
plt.grid(linestyle=':', axis='y')
plt.legend(loc='lower right', prop={'size': 18})

plt.savefig('hidden.pdf')
plt.savefig('hidden.png')

# plt.savefig('proteins.pdf')
# plt.savefig('proteins.png')

