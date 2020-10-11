import numpy as np

arr_1 = "./1.csv"
arr_2 = "./2.csv"

# 加载数据
arr_1 = np.loadtxt(arr_1, delimiter=",", dtype=int)
arr_2 = np.loadtxt(arr_2, delimiter=",", dtype=int)

# 构建全为 0 和 1 的数组
zeros_data = np.zeros((arr_1.shape[0], 1)).astype(int)
ones_data = np.ones((arr_2.shape[0], 1)).astype(int)

# 添加 0 1 数组为列
arr_1 = np.hstack((arr_1, zeros_data))
arr_2 = np.hstack((arr_2, ones_data))

# 竖拼接两组数据
arr_3 = np.vstack((arr_1, arr_2))










