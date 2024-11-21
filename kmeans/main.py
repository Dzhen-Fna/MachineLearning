import matplotlib.pyplot as plt
from numpy import *
import numpy as np

#利用公式计算两个向量p1与p2之间的距离，返回diff
def get_distance(p1, p2):
    diff = (sum(power(p1 - p2, 2)))
    return diff

#1、根据数据文件的路径file_path打开样本数据
#2、将存储的文本信息转换成向量并传出
def load_data(file_path):
    f = open(file_path)
    data = []
    for line in f.readlines():
        row = []
        lines = line.strip().split()
        for x in lines:
            row.append(float(x))
        data.append(row)
    f.close()
    return np.mat(data)

#参数 data:样本数据点集合
#参数 k:已确定的最终聚类的类别数
#return:聚类中心集合my_cluster_center
def random_data(my_data, k):
    sum = np.shape(my_data)[1]  # 特征个数
    print(sum)
    my_cluster_center = np.mat(np.zeros((k, sum)))  # 初始化k个聚类中心
    for j in range(sum):  # 最小值+0-1之间的随机数*变化范围，即得到在最大最小值之间的随机数
        Xmin = np.min(my_data[:, j])  #得到最小值
        Xrange = np.max(my_data[:, j]) - Xmin  #得到随机数的变化范围
        my_cluster_center[:, j] = Xmin * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * Xrange
    print('初始化的聚类中心如下：')
    print(my_cluster_center)
    return my_cluster_center

#参数ata:样本数据点集合
#参数k:聚类中心的个数
#参数cluster_center:传进来的为初始的聚类中心集合
#首先，在数据集中选择K个点作为每个簇的初始中心，接着观察其他的数据，然后将这些数据划分到距离这K个点最近的簇中， 这个过程将数据划分成K个簇，从而完成第一次划分。但实际算法过程中，形成的新簇不一定就是最好的划分，因此在生成的新簇中， 再一次计算每个簇的中心点，然后进行划分，直到每次划分的结果保持不变为止。
def k_means(data, k, cluster_center):
    m, n = np.shape(data)  # m:样本个数；n:特征的维度
    sub_center = np.mat(np.zeros((m, 2)))  # 初始化：每个样本所属的类别（m行n列：共m个样本，第一列为所属类的标号，第二列为最小距离）
    change = 1  # 判断是否重新计算聚类中心
    while change == 1:
        change = 0
        for i in range(m):
            min_distance = np.inf  # 初始样本与聚类中心的最小值为正无穷
            Index = 0  #所属类别
            for j in range(k):
                # 分别计算i到这k个聚类中心的距离，找到距离最近的
                diff = get_distance(data[i,], cluster_center[j,])
                if diff < min_distance:
                    min_distance = diff
                    Index = j
            # 判断所属聚类中心是否发生变化（可能原本就属于这个聚类中心）
            if sub_center[i, 0] != Index:
                change = 1
                sub_center[i,] = np.mat([Index, min_distance])
        # 重新计算聚类中心
        for j in range(k):
            all = np.mat(np.zeros((1, n)))  # all记录所有属于中心j的样本，在n个维度的总和
            r = 0  # 每个类别中的样本个数
            for i in range(m):
                if sub_center[i, 0] == j:  # 属于第j个类别，计算进去
                    all = all + data[i,]
                    r = r + 1
            for a in range(n):
                try:  # sum_all除以本中心的样本个数r，即得到中心
                    cluster_center[j, a] = all[0, a] / r
                except:
                    print("没有样本属于这个聚类中心！")
        # 打印，显示聚类中心的变化过程
        print("聚类中心发生变化如下：")
        print(cluster_center)
    return cluster_center, sub_center

#把数据data写入Myfile_name文件中，并保存
def save_result(Myfile_name, data):
    m, n = np.shape(data)
    f = open(Myfile_name, "w")
    for i in range(m):
        X = []
        for j in range(n):
            X.append(str(data[i, j]))
        f.write("\t".join(X) + "\n")
    f.close()


#输出数据样本点的总个数，并根据数据进行聚类分析，对每一个属性进行聚类，并绘图显示
def draw(point_data, center, sub_center):
    Myfig = plt.figure()
    axes = Myfig.add_subplot(111)

    length = len(point_data)
    print(length)

    for a in range(length):
        if sub_center[a, 0] == 0:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='m', alpha=0.4)
        if sub_center[a, 0] == 1:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='b', alpha=0.4)
        if sub_center[a, 0] == 2:
            axes.scatter(point_data[a, 2], point_data[a, 3], color='g', alpha=0.4)
    for i in range(len(center)):
        axes.scatter(center[i, 2], center[i, 3], color='red', marker='p')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kmeans-JYZ')
    plt.show()


if __name__ == "__main__":
    K = 3  # 预设好聚类中心数k
    print("第一步：导入样本数据")
    sample_data = load_data("seeds_dataset.txt")

    print("第二步：初始化k个聚类中心")
    cluster_center = random_data(sample_data, K)

    print("第三步：K_Means算法")
    cluster_center, sub_center = k_means(sample_data, K, cluster_center)

    print("第四步：保存最终的聚类中心")
    save_result("cluster_center_test.txt", cluster_center)

    print("第五步：保存每个样本所属类别")
    save_result("sub_center_test.txt", sub_center)

    # 画图
    draw(sample_data, cluster_center, sub_center)

