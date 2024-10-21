from torch.cuda import is_available

class Conf_():
    train_acc = []
    valid_acc = []
    test_acc = []
    train_loss = []
    valid_loss = []
    train_timeSplit = []
    valid_timeSplit = []
    test_timeSplit = []
    def __init__(self,lr,use_cuda,model_name,data_name,epochs,batch_size,ratio=0.8):
        self.lr = lr
        self.use_cuda = use_cuda and is_available()
        self.model_name = model_name
        self.data_name = data_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.ratio = ratio
    
    def getPicSize(self):
        '''
        获取图片大小
        '''
        jugge = (self.data_name,self.model_name)
        if jugge == ('MNIST','AlexNet'):
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

