import pandas as pd
from matplotlib import pyplot as plt

file_path = './xx.csv'
df = pd.read_csv(file_path)

# rating,runtime 的分布情况
# 选择图形，直方图
# 准备数据
runtime_data = df['runtime'].value

max_runtime = runtime_data.max()
min_runtime = runtime_data.min()

# 计算组数
num_bin = (max_runtime - min_runtime) // 10

# 设置图形的大小， 绘制，刻度
plt.figure(figsize=(20, 8), dpi=80)
plt.hist(runtime_data, num_bin)

plt.xticks(range(min_runtime, max_runtime+5, 5))

plt.show()
