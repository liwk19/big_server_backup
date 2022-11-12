import numpy as np
import matplotlib.pyplot as plt

origin = [54.47, 56.25, 57.19]
# ngnn = [54.76, 56.57, 57.36]
moe = [54.78, 56.47, 57.39]
x = [1,2,3]
x_ori = [128, 256, 512]

plt.figure(figsize=(12, 6))
plt.xlabel('Hidden size', size=20)
plt.ylabel('Test accuracy (%)', size=20)

plt.scatter(x, origin, label='vanilla', color='blue', linewidth=3)
plt.plot(x, origin, color='blue')
# plt.scatter(x, ngnn, label='NGNN')
# plt.plot(x, ngnn, linewidth=2)
plt.scatter(x, moe, label='GMoE', color='red', linewidth=3)
plt.plot(x, moe, color='red')
plt.xticks(x, x_ori, size=15)
plt.yticks(size=15)
# plt.ylim((87.4, 88.8))
plt.grid(linestyle=':', axis='y')
plt.legend(loc='lower right', prop={'size': 15})
plt.savefig('mag.pdf')
plt.savefig('mag.png')
