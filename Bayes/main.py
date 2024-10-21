from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import operator


# 计算高斯分布密度函数的值
def calculate_gaussian_probability(mean, var, x):
    coeff = (1.0 / (np.math.sqrt((2.0 * np.math.pi) * var)))
    exponent = np.math.exp(-(np.math.pow(x - mean, 2) / (2 * var)))
    c = coeff * exponent
    return c


# 计算均值
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


# 计算方差
def var(list, avg):
    var1 = 0
    for i in list:
        var1 += float((i - avg) ** 2)
    var2 = (np.math.sqrt(var1 / (len(list) * 1.0)))
    return var2


# 朴素贝叶斯分类模型
def Naivebeys(splitData, classset, test):
    classify = []
    for s in range(len(test)):
        c = {}
        for i in classset:
            splitdata = splitData[i]
            num = len(splitdata)
            mu = num + 2
            character = len(splitdata[0]) - 1  # 具体数据集，个数有变
            classp = []
            for j in range(character):
                zi = 1
                if isinstance(splitdata[0][j], (int, float)):
                    numlist = [example[j] for example in splitdata]
                    Mean = averagenum(numlist)
                    Var = var(numlist, Mean)
                    a = calculate_gaussian_probability(Mean, Var, test[s][j])
                else:
                    for l in range(num):
                        if test[s][j] == splitdata[l][j]:
                            zi += 1
                    a = zi / mu
                classp.append(a)
            zhi = 1
            for k in range(character):
                zhi *= classp[k]
            c.setdefault(i, zhi)
        sorta = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
        classify.append(sorta[0][0])
    return classify


# 评估
def accuracy(y, y_pred):
    yarr = np.arry(y)
    y_predarr = np.arry(y_pred)
    yarr = yarr.reshape(yarr.shape[0], -1)
    y_predarr = y_predarr.reshape(y_predarr.shape[0], -1)
    return sum(yarr == y_predarr) / len(yarr)


# 数据处理
def splitDataset(dataSet):  # 按照属性把数据划分
    classList = [example[-1] for example in dataSet]
    classSet = set(classList)
    splitDir = {}
    for i in classSet:
        for j in range(len(dataSet)):
            if dataSet[j][-1] == i:
                splitDir.setdefault(i, []).append(dataSet[j])
    return splitDir, classSet


open('test.txt')
df = pd.read_csv('test.txt')
class_le = LabelEncoder()
dataSet = df.values[:, :]
dataset_train, dataset_test = train_test_split(dataSet, test_size=0.1)
splitDataset_train, classSet_train = splitDataset(dataset_train)
classSet_test = [example[-1] for example in dataset_test]
y_pred = Naivebeys(splitDataset_train, classSet_train, dataset_test)
accu = accuracy(classSet_test, y_pred)
print("Accuracy:", accu)
