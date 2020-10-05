# DataAnalysis

数据分析是用适当的方法对收集来的大量数据进行分析，帮助人们做出判断，以便采取适当的行动

ps：公司内部对产品分析，app 在全国各省下载量如何，条形图折线图展示

数据可能存放在 log 日志中，数据库中


## 一般来说，数据分析的基本过程包括以下几个步骤：

1.提出问题 —— 即我们所想要知道的指标（平均消费额、客户的年龄分布、营业额变化趋势等等）

2.导入数据 —— 把原始数据源导入Jupyter Notebook中（网络爬虫、数据读取等）

3.数据清洗 —— 数据清洗是指发现并纠正数据文件中可识别的错误（检查数据一致性，处理无效值和缺失值等）

4.数据分析 或 构建模型（高级的模型构建会使用机器学习的算法）

5.数据可视化——matplotib库等


## Jupyter Notebook

1.安装：pip install jupyter

2.运行：jupyter notebook，新建文件会创建在打开的目录下


## Matplotlib

Python 底层绘图库，主要做数据可视化图表

安装：pip install matplotlib

- 作用

  - 将数据进行可视化，更直观的呈现

  - 使数据更加客观，更具有说服力

- 基础要点

```python

# pyplot 画图模块
from matplotlib import pyplot as plt

# 数据在 x 轴的位置，是一个可迭代对象
x = range(2, 26, 2)
# 数据在 y 轴的位置，是一个可迭代对象
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 24, 22, 18, 15]

# 设置图片大小
plt.figure(figsize=(20, 8), dpi=80)

# 绘图
plt.plot(x, y)

# 设置 x y 轴的刻度
plt.xticks(x)
plt.yticks(y)

# 保存(png,svg)
plt.savefig("./t1.png")

plt.show()

```

- 设置中文显示

查看 linux 下面支持的字体

fc-list -> 查看支持的字体

fc-list :lang=zh -> 查看支持的中文（冒号前有空格）

```python

import matplotlib
from matplotlib import font_manager

# 方法一: Win 和 Linux 下字体修改
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
matplotlib.rc("font", **font)

# 方法二: 导入 font_manager，实例化字体，在需要的位置使用
my_font = font_manager.FontProperties(fname="字体位置")
plt.xticks(_x[::3], _xtick_labls[::3], rotation=60, fontproperties=my_font)


```
























