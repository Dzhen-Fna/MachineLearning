from params import *
import numpy as np
# 载入数据
data = np.genfromtxt("./data.csv", delimiter=",")
x_data = data[:,0]
y_data = data[:,1]

normal_linear = config_()
normal_linear.gradient_descent_runner(x_data, y_data)
normal_linear.save_data()