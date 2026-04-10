# 实验：灰度、通道与颜色空间

## 实验目标

理解颜色信息如何被重新表达，以及为什么不同任务需要不同颜色空间。

## 对应代码

- Python：`python/convert_to_grayscale.py`
- Python：`python/color_channels.py`
- Python：`python/color_space.py`

## 第一性原理

同一张图像可以有不同表示方式：

- 灰度：关注亮度
- BGR/RGB：关注直接颜色值
- HSV：把颜色、饱和度、亮度分开
- LAB：更接近感知空间

## 实验步骤

1. 将彩图转成灰度图
2. 拆分三个颜色通道
3. 转换到 HSV、LAB、RGB
4. 对比不同空间下的视觉效果

## 预期效果

- 理解为什么灰度图常用于后续边缘和阈值处理
- 理解不同颜色空间适合不同任务
