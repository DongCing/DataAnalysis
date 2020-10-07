### 绘制散点图

```python

from matplotlib import pyplot as plt
from matplotlib import font_manager

# 七天气温
y_3 = [23, 31, 30, 34, 33, 32, 32]

# 天数
x = range(1, 8)

# 绘制散点图
plt.scatter(x, y_3)

```
