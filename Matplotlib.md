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









