from tqdm import tqdm
from math_function import compute_error
import pandas as pd
class config_():
    b_list = []
    k_list = []
    error_list = []
    def __init__(self,epochs=20,lr=0.01,b=0,k=0):
        self.epochs = epochs
        self.lr = lr
        self.b = b
        self.k = k
    def save_data(self):
        print(self.lr)
        df = pd.DataFrame({'b':self.b_list,'k':self.k_list,'error':self.error_list})
        df.to_csv(f'./csv/nomallinear_{self.b_list[0]}_{self.k_list[0]}_{self.epochs}_{self.lr}.csv',index=False)

    def gradient_descent_runner(self,x_data, y_data):
        self.b_list.append(self.b) # 为第一项做特殊处理
        self.k_list.append(self.k)
        m = float(len(x_data))
        print('正在训练模型...')
        for _ in tqdm(range(self.epochs)):
            
            b_grad = 0
            k_grad = 0
        # 计算梯度的总和再求平均
            for j in range(0, len(x_data)):
                b_grad += (1/m) * (((self.k * x_data[j]) + self.b) - y_data[j])
                k_grad += (1/m) * x_data[j] * (((self.k * x_data[j]) + self.b) - y_data[j])
        # 更新b和k
            self.b -= (self.lr * b_grad)
            self.k -= (self.lr * k_grad)
            self.b_list.append(self.b)
            self.k_list.append(self.k)
        self.error_list = [compute_error(self.b_list[i],self.k_list[i],x_data,y_data) for i in range(len(self.k_list))]
    
    def set_error(self,x_data,y_data):
        self.error_list = [compute_error(self.b_list[i],self.k_list[i],x_data,y_data) for i in range(len(self.k_list))]


        
if __name__ == '__main__':
    pass
        
            
