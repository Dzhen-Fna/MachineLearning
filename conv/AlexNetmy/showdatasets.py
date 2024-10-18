import matplotlib.pyplot as plt
from random import randint

def getSingle(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    return images[0], labels[0]


def showSingle(image, label, size):
    image = image.numpy().reshape(size,size)
    label = label.numpy()
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.show()
if __name__ == '__main__':
    pass

