### 中文乱码

```
报错：
findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
Warning (from warnings module):
  File "/home/****/.local/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py", line 214
    font.set_text(s, 0.0, flags=flags)
RuntimeWarning: Glyph 24615 missing from current font.

解决办法：
python代码中字体设置：

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

下载SimHei.ttf，放在/.local/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf 下，
删除 ~/.cache/matplotlib的缓冲目录，里面一个json缓存了字体定义。
重启 IDE
```
