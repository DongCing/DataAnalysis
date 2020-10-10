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

# 绘图，可多次调用 plot 来显示多条折线
# label 结合 legend 绘制折线的图例标注；color 折线颜色参数；linestyle 线条风格；linewidth 线条粗细；alpha 线条透明度
plt.plot(x, y, label="图例", color="orange", linestyle="--", linewidth=5, alpha=0.5)

# 添加图例，只有这里使用 prop 参数来显示中文，loc 图例位置
plt.legend(prop=my_font, loc=0)

# 设置 x y 轴的刻度
plt.xticks(x)
plt.yticks(y)

# 绘制网格,alpha 网格线透明度
plt.grid(alpha=0.4)

# 保存(png,svg)
plt.savefig("./t1.png")

# 添加描述信息
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("温度变化")

plt.show()


# 添加文本注释
# 添加文字（水印）到图中
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

- 多样图形绘制 https://matplotlib.org/gallery/index.html

- 折线图、散点图、柱状图、直方图、条形图、箱线图、饼图等

- 需要知道不同的统计图到底能够表示出什么，从而来选择哪种统计图


## NumPy

一个在 Python 中做科学计算的基础库，重在数值计算，也是大部分 Python 科学计算库的基础库，多用于在大型多维数组上执行数值运算

### NumPy 创建数组（矩阵）

- 创建数组

```python

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array(range(1, 6))
c = np.arange(1, 6)

print(a, b, c)
print(type(a), type(b), type(c))

# 不同于list,数组没有','
# [1 2 3 4 5] [1 2 3 4 5] [1 2 3 4 5]
# <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# dtype 查看数组中存放的数据的类型,返回 NumPy 的数据类型
print(a.dtype)

```

- 数据类型的操作

```python

# 指定创建的数组的数据类型
a = np.array([1, 0, 1, 0], dtype=np.bool)

# 修改数组的数据类型
a.astype("i1")
a.astype(np.int8)

# NumPy 中的小数
# 生成十个小数的数组
b = np.array([random.random() for i in range(10)])
# 保留2位小数
c = np.round(b,2)

# 同样保留小数方法
# round(random.random(), 3)
# "%.2f"%random.random()

```

- 数组的形状

```python

a = np.array([1, 2, 3, 3, 5, 6])

# 查看数组的形状,返回一个元组，元组中有几个数就是几维数组
a.shape

# 修改数组形状
# 2行 3列
b = a.reshape(2, 3)

# 3块 1行 2列
c = a.reshape(3, 1, 2)

# 生成数组时定义形状
d = np.arange(6).reshape(2, 3)

# 将多维数组展开成一维
e = b.flatten()

# 计算广播性质，计算会应用到数组中的每一个数
f = a + 2

# 相同形状数组进行运算，对应位置运算
a + e

# 不同形状的数组，相同维度部分进行运算
# 当长度不匹配时无法进行计算

# 广播原则：若两个数组的后缘维度（trailing dimension，即从末尾开始算起的维度）的轴长度相符或其中一方的长度为1，则认为它们是广播兼容的

# shape为(3, 3, 3)的数组和(3, 2)的数组，不能进行计算
# shape为(3, 3, 2)的数组和(3, 2)的数组，可以进行计算

```

- 轴（axis）

在 NumPy 中可以理解为方向，使用0，1，2...数字表示

对于一维数组，只有一个 0 轴；

对于二维数组（shape(2,2)），有 0 轴和 1 轴

对于三维数组（shape(2,2,3)），有 0，1，2 轴

有了轴的概念之后，计算会更加方便。


- NumPy 读取文件

常用到的作为数据存储的文件类型有：csv，json，vml，hdf等

CSV：Comma-Separated Val，逗号分隔值文件；

显示：表格状态；

源文件：换行和逗号分隔行列的格式化文本，每一行的数据表示一条记录

```python

numpy.loadtxt(fname, dtype=, comments='#', delimiter=None,
              converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)

# fname	被读取的文件名（文件的相对地址或者绝对地址），文件、字符串、产生器、gz或bz2压缩文件

# dtype	指定读取后数据的数据类型，可选，默认 numpy.float

# comments	跳过文件中指定参数开头的行（即不读取）

# delimiter	指定读取文件中数据的分割符

# converters	对读取的数据进行预处理

# skiprows	选择跳过的行数，一般跳过第一行表头

# usecols	指定需要读取的列，索引，元组类型

# unpack	选择是否将数据进行向量输出，True 读入属性将分别写入不同数组变量；False 读入数据只写入一个数组变量，默认 False；
# 将数据按对角线旋转180，行变列，列变行
# 二维数组转置方法：arr.transpose(), arr.T, arr.swapaxes(1, 0)

# encoding	对读取的文件进行预编码

```

- NumPy 保存数据
```python

numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', 
              header='', footer='', comments='# ', encoding=None)

# fname ： 文件名或文件句柄，如果文件名结束.gz，文件将自动以压缩gzip格式保存。 loadtxt透明地理解gzip文件。

# X ： 1D或2D array_like，要保存到文本文件的数据。

# fmt ： str或strs序列，可选
# 单个格式（％10.5f），格式序列或多格式字符串，例如“迭代％d - ％10.5f”，在这种情况下，将忽略分隔符。对于复杂的X，fmt的合法选项是：
# 单个说明符，fmt ='％.4e'，导致数字格式为'（％s +％sj）'％（fmt，fmt）
# 一个完整的字符串，指定每个实部和虚部，例如 '％.4e％+.4ej％.4e％+.4ej％.4e％+.4ej'为3列
# 一个说明符列表，每列一个 - 在这种情况下，实部和虚部必须有单独的说明符，例如['％.3e +％.3ej'，'（％.15e％+.15ej）'] 2列

# delimiter 分隔符 ： str，可选，分隔列的字符串或字符。

# newline 换行符 ： str，可选，字符串或字符分隔线。

# header ： str，可选，将在文件开头写入的字符串。

# footer 页脚 ： str，可选，将写在文件末尾的字符串。

# comments 评论 ： str，可选，将附加到header和footer字符串的字符串，以将其标记为注释。默认值：'＃'，正如预期的那样 numpy.loadtxt。

# encoding ： {None，str}，可选
# 用于编码输出文件的编码。不适用于输出流。如果编码不是'bytes'或'latin1'，您将无法在NumPy版本<1.14中加载该文件。默认为'latin1'。

```






















