import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from math_function import compute_error
from os import listdir # 用于获取文件名

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
def get_csv_name(csv_path='./csv'):
    '''
    
    '''
    files_name = listdir(csv_path)
    return [i for i in files_name if i.endswith('.csv')]

def plot_surf(csv_file_name):
    csv = np.genfromtxt('./csv/'+csv_file_name, delimiter=',',skip_header=1)
    CONSTANT = np.genfromtxt('data.csv', delimiter=',')
    B = np.linspace(
        min(csv[:,0]),
        max(csv[:,0]),
        100
    )
    K = np.linspace(
        min(csv[:,1]),
        max(csv[:,1]),
        100
    )
    B, K = np.meshgrid(B, K)
    Z = compute_error(B, K, CONSTANT[:,0], CONSTANT[:,1])
    # 创建一个3D曲面，用于表示梯度
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(B, K, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # 设置坐标轴范围
    # ax.set_xlim(np.floor(min(csv[:,0])), np.ceil(max(csv[:,0])))
    # ax.set_ylim(np.floor(min(csv[:,1])), np.ceil(max(csv[:,1])))
    # ax.set_zlim(np.floor(min(csv[:,2])), np.ceil(max(csv[:,2])))
    # 设置z轴的刻度
    ax.zaxis.set_major_locator(LinearLocator(1))
    # 设置坐标轴的刻度条格式
    ax.xaxis.set_major_formatter('{x:.02f}')
    ax.yaxis.set_major_formatter('{x:.02f}')
    ax.zaxis.set_major_formatter('{x:.02f}')
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # 设置标题
    ax.set_title('损失函数的表示')
    # 设置坐标轴标签
    ax.set_xlabel('B')
    ax.set_ylabel('K')
    ax.set_zlabel('Error')
    plt.show()

def plot_linear(csv_file_name):
    data = np.genfromtxt('data.csv', delimiter=',')
    x = data[:,0]
    y = data[:,1]
    b = np.genfromtxt('./csv/'+csv_file_name, delimiter=',',skip_header=1)[-1,0]
    k = np.genfromtxt('./csv/'+csv_file_name, delimiter=',',skip_header=1)[-1,1]
    plt.plot(x, y, 'b.')
    plt.plot(x, k*x+b,'r')
    plt.show()

def plot_error_grading(csv_file_name):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    data = np.genfromtxt('./csv/'+csv_file_name, delimiter=',',skip_header=1)
    ax.stem(
        data[:,0], 
        data[:,1],
        data[:,2]
        )
    ax.set_xlabel('B')
    ax.set_ylabel('K')
    ax.set_zlabel('Error')
    plt.show()
for idx,ele in enumerate(get_csv_name()):
    print(f'{idx} : {ele}')
choose = int(input('选择需要绘图的文件\n'))
plot_surf(get_csv_name()[choose])
plot_linear(get_csv_name()[choose])
plot_error_grading(get_csv_name()[choose])

