- 文件说明
    - KNN_sk.py -- > 使用sklearn库实现KNN算法
    - KNN_self.py -- > 使用KNN算法实现MNIST数据集预测
    - show_img.py --> 显示图片
    - gauss.csv --> 高斯数据集

- 文件依赖关系 \
![K-means算法的文件结构](./mdpic/files_tree.png)


# 使用KNN算法实现MNIST数据集预测
## 1. 数据集介绍
- mnist_x.txt


```math
X =\begin{bmatrix}
x_{0,0} & x_{0,1} & \cdots & x_{0,783} \\
x_{1,0} & x_{1,1} & \cdots & x_{1,783} \\
x_{2,0} & x_{2,1} & \cdots & x_{2,783} \\
\cdots & \cdots & \cdots & \cdots \\
x_{999,0} & x_{999,1} & \cdots & x_{999,783} \\
\end{bmatrix} 
```

X中每一列代表一张图片，共1000张图片，每张图片有784个像素点，像素点通过0与1表示，其中0代表白色像素点，1代表黑色像素点，因此X是一个1000*784的矩阵。


- mnist_y.txt
```math
Y=\begin{bmatrix}
y_{0} \\
y_{1} \\
y_{2} \\
\cdots \\
y_{999} \\
\end{bmatrix}
```
Y中每一行代表一张图片的标签，共1000张图片，因此Y是一个1000*1的矩阵。

- 训练集以及测试集划分
```math
X_{train} : X_{test} = 0.8 : 0.2 \\
Y_{train} : Y_{test} = 0.8 : 0.2
```


## 2. KNN算法介绍
- KNN算法是一种基于实例的学习，即它没有显式的训练过程，而是直接使用训练数据来预测测试数据。KNN算法的核心思想是：如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。KNN算法的步骤如下：

1. 计算待分类样本与所有已知样本的距离。
2. 找出距离最近的K个已知样本。
3. 统计这K个已知样本中各个类别的数量。
4. 将待分类样本归为数量最多的类别。

## 3. KNN算法实现

![K-means算法的文件结构](./mdpic/KNN_self.png)

# KNN_sk
- 数据集介绍 _gauss.csv_
