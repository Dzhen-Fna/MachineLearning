from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
import numpy as np
import showdatasets

def get_data(conf):
    ratio = conf.ratio
    img_size = conf.getPicSize()
    batch_size = conf.batch_size
    data_train = MNIST(root='../data',  # 60,000
                    train=True, 
                    transform=transforms.Compose(
                        [transforms.Resize(img_size),  
                            transforms.ToTensor()]
                    ))
    data_train, data_valid = random_split(data_train, [round(ratio*len(data_train)), round((1-ratio)*len(data_train))]) # 后续更换CIFA需要注意
    data_test = MNIST(root='../data', # 10,000
                    train=False, 
                    transform=transforms.Compose(
                        [transforms.Resize(img_size),  
                            transforms.ToTensor()]
                    ))

    data_train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        )# 删除shuffle=True,使用random_split代替
    data_valid_loader = DataLoader(dataset=data_valid,
                                       batch_size=batch_size)
    data_test_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=True,
    )
    return data_train_loader,data_valid_loader,data_test_loader


# img,lable = showdatasets.getSingle(data_train_loader)
# showdatasets.showSingle(img,lable, 224)


