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
