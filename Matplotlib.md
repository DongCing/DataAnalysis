### 绘制散点图

```python

from matplotlib import pyplot as plt
from matplotlib import font_manager

# 

# 七天气温
y_1 = [23, 31, 30, 34, 33, 32, 32]
y_2 = [30, 30, 29, 34, 33, 33, 32]

# 天数
x_1 = range(1, 8)
x_2 = range(8, 15)

# 设置图形大小
plt.figure(figsize(20, 8), dpi=80)

# 绘制散点图
plt.scatter(x_1, y_1)
plt.scatter(x_2, y_2)

# 调整刻度
_x = list(x_1) + list(x_2)
_xtick_labels = ["1月{}日".format(i) for i in x_1]
_xtick_labels += ["2月{}日".format(i-7) for i in x_2]
plt.xticks(_x, _xtick_labels, fontproperties = my_font)

```
