from torchvision.models import AlexNet
from torch import nn
from torch import cuda
import torch

AlexNet_MNIST= AlexNet(10)
AlexNet_MNIST.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
if __name__ == '__main__':
    model = AlexNet_MNIST
    model = model.cuda()
    random_tensor = torch.rand(1, 1, 224, 224).cuda()
    output = model(random_tensor)
    print(output.shape)
    