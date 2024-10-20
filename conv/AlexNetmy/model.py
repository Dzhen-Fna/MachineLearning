from torchvision.models import AlexNet, googlenet, resnet18,resnet50, vgg16,vgg19
from torchvision.models.googlenet import BasicConv2d

from torch import nn,cuda,rand,optim
from utils import Conf_
def get_model(conf):
    judge = conf.getModel_Data_names()
    if judge == ('AlexNet','MNIST'):
        model_ = AlexNet(10)
        model_.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    elif judge == ('GoogleLeNet','MNIST'):
        model_= googlenet()
        model_.conv1 = BasicConv2d(1, 64,kernel_size=7, stride=2, padding=3)
        model_.fc = nn.Linear(in_features=1024, out_features=10, bias=True)
    elif judge == ('ResNet18','MNIST'):
        model_= resnet18()
        model_.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_.fc = nn.Linear(512, 10, bias=True)
    elif judge == ('ResNet50','MNIST'):
        model_= resnet50()
        model_.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_.fc = nn.Linear(2048, 10, bias=True)
    elif judge == ('VGG16','MNIST'):
        model_= vgg16()
        model_.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model_.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
    elif judge == ('VGG19','MNIST'):
        model_= vgg19()
        model_.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model_.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
    return model_.cuda() if conf.use_cuda else model_

def get_criterionAndopti(conf,model_parameters):
    '''
        获取损失函数
        获取优化器
    '''
    judge = conf.getModel_Data_names()
    if  judge == ('AlexNet','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)
    elif judge == ('GoogleLeNet','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)
    elif judge == ('ResNet18','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)
    elif judge == ('ResNet50','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)
    elif judge == ('VGG16','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)
    elif judge == ('VGG19','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_parameters, lr=conf.lr)

    return criterion,optimizer
if __name__ == '__main__':
    conf = Conf_(
    lr=0.01,
    use_cuda=False,
    model_name='AlexNet',
    data_name='MNIST',
    epochs=2,
    batch_size=100,
    ratio=0.8)
    model = get_model(conf)
    print(AlexNet(10))
    