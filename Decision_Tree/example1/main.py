import pandas as pd
from math import log2
from pylab import *
import matplotlib.pyplot as plt


def load_dataset():
    # 数据集文件所在位置
    path = "xigua.csv"
    data = pd.read_csv(path, header=0, encoding='gbk')
    dataset = []
    for a in data.values:
        dataset.append(list(a))
    # 返回数据列表
    attribute = list(data.keys())
    # 返回数据集和每个维度的名称
    return dataset, attribute

dataset,attribute = load_dataset()
print(attribute,dataset)


def calculate_info_entropy(dataset):
    # 记录样本数量
    n = len(dataset)
    # 记录分类属性数量
    attribute_count = {}
    # 遍历所有实例，统计类别出现频次
    for attribute in dataset:
        # 每一个实例最后一列为类别属性，因此取最后一列
        class_attribute = attribute[-1]
        # 如果当前类标号不在label_count中，则加入该类标号
        if class_attribute not in attribute_count.keys():
            attribute_count[class_attribute] = 0
        # 类标号出现次数加1
        attribute_count[class_attribute] += 1
    info_entropy = 0
    for class_attribute in attribute_count:
        # 计算该类在实例中出现的概率
        p = float(attribute_count[class_attribute]) / n
        info_entropy -= p * log2(p)
    return info_entropy


def split_dataset(dataset,i,value):
    split_set = []
    for attribute in dataset:
        if attribute[i] == value:
            # 删除该维属性
            reduce_attribute = attribute[:i]
            reduce_attribute.extend(attribute[i+1:])
            split_set.append(reduce_attribute)
    return split_set


def calculate_attribute_entropy(dataset,i,values):
    attribute_entropy = 0
    for value in values:
        sub_dataset = split_dataset(dataset,i,value)
        p = len(sub_dataset) / float(len(dataset))
        attribute_entropy += p*calculate_info_entropy(sub_dataset)
    return attribute_entropy


def calculate_info_gain(dataset,info_entropy,i):
    # 第i维特征列表
    attribute = [example[i] for example in dataset]
    # 转为不重复元素的集合
    values = set(attribute)
    attribute_entropy = calculate_attribute_entropy(dataset,i,values)
    info_gain = info_entropy - attribute_entropy
    return info_gain


def split_by_info_gain(dataset):
    # 描述属性数量
    attribute_num = len(dataset[0]) - 1
    # 整个数据集的信息熵
    info_entropy = calculate_info_entropy(dataset)
    # 最高的信息增益
    max_info_gain = 0
    # 最佳划分维度属性
    best_attribute = -1
    for i in range(attribute_num):
        info_gain = calculate_info_gain(dataset,info_entropy,i)
        if(info_gain > max_info_gain):
            max_info_gain = info_gain
            best_attribute = i
    return best_attribute


def create_tree(dataset,attribute):
    # 类别列表
    class_list = [example[-1] for example in dataset]
    # 统计类别class_list[0]的数量
    if class_list.count(class_list[0]) == len(class_list):
        # 当类别相同则停止划分
        return class_list[0]
    # 最佳划分维度对应的索引
    best_attribute = split_by_info_gain(dataset)
    # 最佳划分维度对应的名称
    best_attribute_name = attribute[best_attribute]
    tree = {best_attribute_name:{}}
    del(attribute[best_attribute])
    # 查找需要分类的特征子集
    attribute_values = [example[best_attribute] for example in dataset]
    values = set(attribute_values)
    for value in values:
        sub_attribute = attribute[:]
        tree[best_attribute_name][value] =create_tree(split_dataset(dataset,best_attribute,value),sub_attribute)
    return tree
tree = create_tree(dataset,attribute)
print(tree)


# 定义划分属性节点样式
attribute_node = dict(boxstyle="round", color='#00B0F0')
# 定义分类属性节点样式
class_node = dict(boxstyle="circle", color='#00F064')
# 定义箭头样式
arrow = dict(arrowstyle="<-", color='#000000')


# 计算叶结点数
def get_num_leaf(tree):
    numLeafs = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += get_num_leaf(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的层数
def get_depth_tree(tree):
    maxDepth = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + get_depth_tree(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制文本框
def plot_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(tree, parentPt, nodeTxt):
    numLeafs = get_num_leaf(tree)
    depth = get_depth_tree(tree)
    firstStr = list(tree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plot_text(cntrPt, parentPt, nodeTxt)  #在父子结点间绘制文本框并填充文本信息
    plotNode(firstStr, cntrPt, parentPt, attribute_node)  #绘制带箭头的注释
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, class_node)
            plot_text((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

# 绘制箭头
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow)

# 绘图
def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_num_leaf(tree))
    plotTree.totalD = float(get_depth_tree(tree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(tree, (0.5, 1.0), '')
    plt.show()


#指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 绘制决策树
createPlot(tree)

