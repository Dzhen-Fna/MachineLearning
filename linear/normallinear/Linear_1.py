import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 1111
# 载入数据
data = np.genfromtxt("./data.csv", delimiter=",")
x_data = data[:,0]
y_data = data[:,1]

b_piont = []
k_piont = []
e_piont = []
# 学习率learning rate
lr = 0.01
# 截距
b = 0
# 斜率
k = 0
# 最大迭代次数
epochs = 100

# 最小二乘法
def compute_error(b, k, x_data, y_data):
    """
    params:
        b: 截距
        k: 斜率
        x_data: x数据
        y_data: y数据
    return:
        平均误差
    """
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0

# 训练用的函数
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    '''
    params:
        x_data: x数据
        y_data: y数据
        b: 截距
        k: 斜率
        lr: 学习率
        epochs: 迭代次数
    '''
    # 计算总数据量
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        b_piont.append(b)
        k_piont.append(k)
        e_piont.append(compute_error(b, k, x_data, y_data))

        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        # 更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
    return b, k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))


#画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k*x_data + b, 'r')
plt.show()



from matplotlib import cm
from matplotlib.ticker import LinearLocator

# 创建一个3D图形
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
# 创建X和Y的坐标
B = np.arange(0, 0.25, 0.01)
K = np.arange(0, 2.5, 0.1)
# 创建X和Y的网格
B, K = np.meshgrid(B, K)
# 计算R的值
Z = compute_error(B, K, x_data, y_data)


# 绘制曲面图
surf = ax.plot_surface(B, K, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# 自定义z轴
ax.set_zlim(0, 100)
# 设置z轴的刻度
ax.zaxis.set_major_locator(LinearLocator(1))
# 设置z轴的刻度格式
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel('B')
ax.set_ylabel('K')
ax.set_zlabel('Error')

# 添加一个颜色条，将值映射到颜色
fig.colorbar(surf, shrink=0.5, aspect=5)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.stem(b_piont, k_piont, e_piont)
ax.set_xlabel('B')
ax.set_ylabel('K')
ax.set_zlabel('Error')

# 显示图形
plt.show()

