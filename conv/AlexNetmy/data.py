from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import showdatasets

data_train = MNIST(root='../data', 
                   train=True, 
                   transform=transforms.Compose(
                       [transforms.Resize((224,224)), #原始数据集28*28 --> 32*32 LeNet
                        transforms.ToTensor()]
                   ))

data_test = MNIST(root='../data', 
                   train=False, 
                   transform=transforms.Compose(
                       [transforms.Resize((224,224)), #原始数据集28*28 --> 32*32 LeNet
                        transforms.ToTensor()]
                   ))

data_train_loader = DataLoader(
    data_train,
    batch_size=64,
    shuffle=True,)
data_test_loader = DataLoader(
    data_test,
    batch_size=64,
    shuffle=True,
)


#### 查看数据集 ####
img,lable = showdatasets.getSingle(data_train_loader)
showdatasets.showSingle(img,lable, 224)


