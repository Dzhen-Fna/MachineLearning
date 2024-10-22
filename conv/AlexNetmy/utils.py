from torch.cuda import is_available # 判断是否有GPU
from pandas import DataFrame as DF # 用于保存数据
from itertools import product # 生成实验参数

class Conf_():

    def __init__(self,lr,use_cuda,model_name,data_name,epochs,batch_size,ratio=0.8):
        self.lr = lr
        self.use_cuda = use_cuda and is_available()
        self.model_name = model_name
        self.data_name = data_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.ratio = ratio

        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.train_timeSplit = []
        self.valid_timeSplit = []
        self.test_timeSplit = []
    
    def getPicSize(self):
        '''
        获取图片大小，用于绘图，与训练函数无关
        '''
        judge = (self.data_name,self.model_name)
        if judge == ('MNIST','AlexNet'):
            return (224,224)
        elif judge == ('MNIST','GoogleLeNet'):
            return (224,224)
        elif judge == ('MNIST','ResNet18'):
            return (224,224)
        elif judge == ('MNIST','ResNet50'):
            return (224,224)
        elif judge == ('MNIST','VGG16'):
            return (224,224)
        elif judge == ('MNIST','VGG19'):
            return (224,224)
        
    def getModel_Data_names(self):
        '''
        获取模型和数据集名
        '''
        return (self.model_name,self.data_name)
    def appendTrainAcc(self,acc):
        self.train_acc.append(acc)

    def appendValidAcc(self,acc):
        self.valid_acc.append(acc)

    def appendTestAcc(self,acc):
        self.test_acc.append(acc)

    def appendTrainLoss(self,loss):
        self.train_loss.append(loss)

    def appendValidLoss(self,loss):
        self.valid_loss.append(loss)

    def appendTrainTime(self,time):
        self.train_timeSplit.append(time)

    def appendValidTime(self,time):
        self.valid_timeSplit.append(time)

    def appendTestTime(self,time):
        self.test_timeSplit.append(time)
    def showTrainStatus(self):
        print("train_acc:",self.train_acc[-1])
        print("train_loss:",self.train_loss[-1])
        print("train_time:",self.train_timeSplit[-1])
    def showValidStatus(self):
        print("valid_acc:",self.valid_acc[-1])
        print("valid_loss:",self.valid_loss[-1])
        print("valid_time:",self.valid_timeSplit[-1])
    def showTestStatus(self):
        print("test_acc:",self.test_acc[-1])
        print("test_time:",self.test_timeSplit[-1])

    def getSaveName(self,fileType):
        '''
        获取保存文件名
        '''
        params = [self.data_name,self.model_name,self.epochs,self.lr,self.use_cuda,self.lr,self.batch_size]

        return '_'.join([str(i) for i in params]) + '.' + fileType
    def saveCSV(self,dir_):
        df = DF({'train_acc':self.train_acc,'valid_acc':self.valid_acc,'test_acc':self.test_acc,'train_loss':self.train_loss,'valid_loss':self.valid_loss,'train_time':self.train_timeSplit,'valid_time':self.valid_timeSplit,'test_time':self.test_timeSplit})
        df.to_csv(dir_ + self.getSaveName('csv'),encoding='utf-8')
    
    def clear(self):
        '''
        用于异常值处理，暂未使用
        '''
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.train_timeSplit = []
        self.valid_timeSplit = []
        self.test_timeSplit = []

# 检测可以运行的参数
model_names = ['AlexNet','GoogleLeNet','ResNet18','LeNet','VGG16']
lrs = [0.0001,0.001,0.01,0.1]
batch_sizes = [32,64,512]
use_cuda = [True,False]

prcs = product(model_names,batch_sizes,use_cuda,lrs)
conf_list = [Conf_(lr=i[3],batch_size=i[1],use_cuda=i[2],model_name=i[0],data_name='MNIST',epochs=50,ratio=0.8) for i in prcs]
if __name__ == '__main__':
    for i in conf_list:
        print(i.getSaveName(''))


