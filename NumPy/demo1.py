import numpy as np

arr_1 = "./1.csv"
arr_2 = "./2.csv"

# 加载数据
arr_1 = np.loadtxt(arr_1, delimiter=",", dtype=int)
arr_2 = np.loadtxt(arr_2, delimiter=",", dtype=int)
