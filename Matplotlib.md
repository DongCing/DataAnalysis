### 前端 JS 绘图框架 ECharts <= 后端准备数据；Plotly，Seaborn 画图

### 总结
- matplotlib.plot(x, y)，折线图
- matplotlib.bar(x, y)，条线图
- matplotlib.scatter(x, y)，散点图
- matplotlib.hist(data, bins, normed)，直方图
- xticks 和 yticks 的设置
- label 和 titile.grid 的设置
- 绘图的大小和保存图片

### 使用流程：1.明确问题  2.选择图形的呈现方式  3.准备数据  4.绘图和图形完善

### 绘制散点图

```python

from matplotlib import pyplot as plt
from matplotlib import font_manager

# fc-list :lang=zh 查看中文字体文件位置
my_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc")

# 七天气温
y_1 = [23, 31, 30, 34, 33, 32, 32]
y_2 = [30, 30, 29, 34, 33, 33, 32]

# 天数
x_1 = range(1, 8)
x_2 = range(8, 15)

# 设置图形大小
# plt.figure(figsize=(20, 8), dpi=80)

# scatter 绘制散点图
plt.scatter(x_1, y_1)
plt.scatter(x_2, y_2)

# 调整刻度
_x = list(x_1) + list(x_2)
_xtick_labels = ["1月{}日".format(i) for i in x_1]
_xtick_labels += ["2月{}日".format(i-7) for i in x_2]
plt.xticks(_x, _xtick_labels, fontproperties=my_font)

plt.show()

```

### 绘制条形图

```python

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

a = ["我和我的家乡", "姜子牙", "夺冠"]
b = [10.14, 7.72, 3.58]

# 绘制条形图,根据 x 轴数据个数,绘制数据条数
# bar 绘制竖条形图
plt.bar(range(len(a)), b, width=0.3)
# barh 绘制横条形图,height 表示宽度,刻度也要改成 y 轴的
plt.bar(range(len(a)), b, height=0.3)

# 设置字符串到 x轴刻度
plt.xticks(range(len(a)), a,)

# 绘制网格
plt.grid()

plt.show()

```

### 绘制直方图

将数据（列表）分成多少组进行统计，适合没有进行统计过的数据

组数要适当，一般 组数 = 极差 / 组距

组距：每个小组的两个端点的距离

极差：最大值 - 最小值

```python

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

a = [10, 20, 30, 40, 50, 11, 21, 31, 24, 15, 13, 41, 5, 48, 40]

# 计算组数
# 组距
d = 5
num_bins = (max(a)-min(a)) // d

plt.figure(figsize=(20, 8), dpi=80)

# 绘制直方图,density在 y 轴显示比率
plt.hist(a, num_bins, density=True)

# 设置 x 轴的刻度
plt.xticks(range(min(a), max(a)+d, d))

plt.grid()

plt.show()

```

### 当数据已经统计，可以用条形图模拟直方图

```python

from matplotlib import pyplot as plt

interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]
width = [5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 30, 60]
quantity = [836, 2737, 3723, 3926, 3596, 1438, 3273, 642, 824, 613, 215, 47]

plt.figure(figsize=(20, 8), dpi=80)

# width=1 条形图之间没有空隙
plt.bar(range(12), quantity, width=1)

# 设置 x 轴的刻度,额外添加最后一个刻度
_x = [i-0.5 for i in range(13)]
_xtick_labels = interval + [150]
plt.xticks(_x, _xtick_labels)

plt.grid()
plt.show()

```











