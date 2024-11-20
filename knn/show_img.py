# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
import numpy as np
import matplotlib.pyplot as plt
x= np.genfromtxt('mnist_x.txt', delimiter=' ')
img1 = x[0,:]
img1 = np.reshape(img1, (28,28))

plt.imshow(img1,cmap='gray') # 查看数据集