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

- NumPy 中的索引和切片，取值

只获取数组中的某一行，或者某一列。操作同 Python 中列表的操作
```python

# 取索引为 2 的行
arr[2]

# 取多行，索引为 2 及以后行
arr[2:]

# 取不连续的多行，索引为 2 8 10 的行，加一 []
arr[[2, 8, 10]]

# 取列
# 取行取列通用写法，‘,’前为行，后为列；下例表示取第一行和所有列
arr[1, :]
arr[2:, :]
arr[[2, 8, 10], :]

# 取连续的多列,下例表示取所有行的索引为 2 及以后的列
arr[:, 2:]

# 取多行多列
arr[2:5, 1:4]

# 取多个不相邻的点(0,0)(2,1)
arr[[0,2], [0,1]]

```

- NumPy 修改数值

取值然后赋值

```python

# 修改 arr 中小于 10 的数值
arr[arr<10] = 0

```

- NumPy 中的三元运算符

```python

# numpy.where(条件, True, False)
numpy.where(arr<=10, 0, 20)

```

- NumPy 中的 clip （裁剪）

```python

# arr 中小于 10 的数替换为 10，大于 20 的数替换为 20
arr.clip(10, 20)
numpy.clip(arr, 10, 20)

```

- NumPy 数组的拼接（对应分割）
```python

# 竖直拼接（vertically），向下拼接到行，列数不变
numpy.vstack((arr1, arr2))

# 水平拼接（horizontally），向右拼接到列，行数不变
numpy.hstack((arr1, arr2))

```

- NumPy 数组的行列交换
```python

# 行交换，选中2 3行，进行交换
arr[[1, 2], :] = arr[[2, 1], :]

# 列交换，选中2 3列，进行交换
arr[：, [1, 2]] = arr[：, [2, 1]]

```

- NumPy 其他方法
  
  - 获取最大值最小值的位置：
  
    - numpy.argmax(t, axis=0)
    
    - numpy.argmin(t, axis=1)
  
  - 创建一个全为 0 的数组：numpy.zeros((3, 4))
  
  - 创建一个全为 1 的数组：numpy.ones((3, 4))
  
  - 创建一个对角线为 1 的正方形数组：numpy.eye(3)

  - 生成随机数（生成随机数1-10，三行四列数组）：numpy.random.randint(1, 10, (3, 4))

- NumPy 中 copy 和 view

  - a = b 完全不复制，a 和 b 相互影响
  
  - a = b[:]，视图的操作，一种切片，会创建新的对象 a，但是 a 的数据完全由 b 保管，他们两个的数据变化是一致的
  
  - a = b.copy()，复制，a 和 b 互不影响
  
- NumPy 中的 nan 和 inf

  - nan(NAN, NAN)：not a number 表示不是一个数字
  
    - 什么时候 NumPy 中会出现 nan
    
      - 读取文件为 float 的时候，如果有缺失，就会出现 nan
    
      - 进行一个不合适的计算的时候（比如无穷大 inf 减去无穷大）
      
    - 注意点
      
      - 两个 nan 是不相等的：numpy.nan != numpy.nan
      
      - 判断数组中 nan 的个数：numpy.count_nonzero(t != t)
      
      - 判断一个数字是否为 nan：numpy.isnan(a)
      
      - 把 nan 替换为 0 ：t[np.isnan(t)] = 0
      
      - nan 和任何值计算都为 nan 
  
  - inf(-inf, inf)：infinity，inf 表示正无穷，-inf 表示负无穷
  
    - 什么时候会出现 inf
    
      - 一个数字除以 0

- NumPy 中常用统计函数

  - 求和：arr.sum(axis=None)
  
  - 均值：arr.mean(a, axis=None) 受离群点的影响较大
  
  - 中值：numpy.median(arr, axis=None)
  
  - 最大值：arr.max(axis=None)
  
  - 最小值：arr.min(axis=None)
  
  - 极值：numpy.ptp(arr, axis=None) 最大值和最小值之差
  
  - 标准差：arr.std(axis=None) 一组数据平均值分散程度的一种度量


## Pandas

NumPy 能够处理数值型数据

Pandas 能够处理更多的数据类型，如字符串，时间序列等

- Pandas 的常用数据类型

  - Series 一维，带标签数组（带有数组的索引）
    - Series 对象本质上由两个数组构成
      - 一个数组构成对象的键(index, 索引)
      - 一个数组构成对象的值(values)
      
  - DataFrame 二维，Series 容器
  
```python

import pandas as pd
import numpy as np
import string

# Series 一维，带标签数组（带有数组的索引）
# 参数 index 可以制定索引,需和数据索引长度一致
t1 = pd.Series([1, 3, 5])
t2 = pd.Series(np.arange(10), index=list(string.ascii_uppercase[:10]))

# 通过字典创建 Series,默认索引就是字典的键
# 可以重新指定索引,若新索引能对应字典的键,取对应值;否则值为 NaN
a = {string.ascii_uppercase[i]: i for i in range(10)}
t3 = pd.Series(a, index=[1, 'A', 3, 4, 5, 6, 7, 8, 9, 0, 10])

# NumPy 中 NaN 为float,Pandas 会自动根据数据类型更改 Series 的 dtype 类型
# 修改 dtype 和 numpy 的方法一样
t3.astype(float)

print(t3, type(t3))

```

- Pandas：Series 切片和索引

  - 切片：直接传入 start end 或者步长
  
  - 索引：一个的时候直接传入序号或者 index，多个时传入序号或者 index 的列表

- Pandas：读取外部数据

```python
import pandas as pd

# Pandas 读取 csv 中的 数据,设置编码格式
# pd.read_ excel json html...
df = pd.read_csv("test_data.csv", encoding='gb18030')

print(df, type(df))

# df 数据类型 DataFrame

```

- Pandas：DataFrame 数据类型

```python

import pandas as pd
import numpy as np

# 创建 DataFrame 对象
# 既有行索引(index, 0轴, axis=0),又有列索引(columns, 1轴, axis=1)
df = pd.DataFrame(np.arange(12).reshape(3, 4), index=list('abc'), columns=list('WXYZ'))
print(df)

# 传入字典数据,列是字典的键
t = {'name': ['Tom', 'Rose'], 'age': [12, 13]}
t2 = pd.DataFrame(t)
print(t2)

# 没有的值为 NaN
d = [{'name': 'tom', 'age': 12, 'tel': 100}, {'name': 'rose', 'age': 11}, {'name': 'jack', 'tel': 110}]
d2 = pd.DataFrame(d)
print(d2)

f = {'name': {'age': 10}}
print(f['name']['age'])

```

  - DataFrame 的基础属性，df 是定义的 DataFrame 对象
    
    df.shape 行数 列数
    
    df.dtypes 列数据类型
    
    df.ndim 数据维度
    
    df.index 行索引
    
    df.columns 列索引
    
    df.values 对象值，二维 ndarray 数组
  
  - DataFrame 整体情况查询
  
    df.head(3) 显示头部几行，默认 5 行
    
    df.tail(3) 显示末尾几行，默认 5 行
    
    df.info() 相关信息概览：行数，列数，列索引，列非空值个数，列类型，内存占用
    
    df.describe() 快速综合统计结果：计数，均值，标准差，最大值，四分位数，最小值
    
  - 排序，根据哪一列，升序降序
   
    df.sort_values(by='列名', ascending=False)
    
    df_sorted[:100] 取前 100 行
    
    df_sorted['列名'] 取某列
    
    df_sorted[:100]['列名'] 取 100 行的某列
    
  - 取值 df.loc 通过标签索引行数据(自定义的索引)

    df.loc['行索引', '列索引'] 取某一个值
    
    df.loc[['', ''], ['', '']] 取多行多列
    
    df.loc[[''： ''], [''： '']] 取多行多列，会取到冒号后面的数据
  
  - 取值 df.iloc 通过位置获取行数据
    
    df.iloc[1:3, [2,3]] 获取获取行索引（1 2） 列索引（2 3）的数据
    
    df.iloc[1:3, 1:3]
    
  - 赋值更改数据
  
    df.loc['A', 'Y'] = 100
    
    df.iloc[1:2, 0:2] = np.nan
    
- Pandas：布尔索引

  df[ df['列'] > 500]
    
    
    



