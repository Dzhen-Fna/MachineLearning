from torchvision.models import AlexNet
from torch import nn,cuda,rand,optim

def get_model(conf):
    judge = conf.getModel_Data_names()
    if judge == ('AlexNet','MNIST'):
        model_ = AlexNet(10)
        model_.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        model_ = model_.cuda() if conf.use_cuda else model_
    return model_

def get_criterionAndopti(conf,model_parameters):
    '''
        获取损失函数
        获取优化器
    '''
    judge = conf.getModel_Data_names()
    if  judge == ('AlexNet','MNIST'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_parameters, lr=conf.lr)
    return criterion,optimizer
if __name__ == '__main__':
    model = get_model(('AlexNet','MNIST'))
    random_tensor = rand(1, 1, 224, 224)
    output = model(random_tensor)
    print(output)
    print(output.shape)
    