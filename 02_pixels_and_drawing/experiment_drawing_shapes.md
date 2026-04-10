# 实验：绘制几何图形

## 实验目标

通过直线、矩形和圆，理解 OpenCV 绘图 API 的输入参数与结果之间的关系。

## 对应代码

- Python：`python/draw_line.py`
- Python：`python/draw_rectangle.py`
- Python：`python/draw_circle.py`

## 实验步骤

1. 在空白画布上绘制一条线
2. 修改起点、终点和 thickness
3. 绘制矩形并尝试 `FILLED`
4. 绘制圆并改变半径

## 需要观察的点

- thickness 对视觉结果的影响
- `FILLED` 与普通边框的区别
- 不同图形共享的坐标逻辑

## 预期效果

- 能独立写出基础绘图代码
- 能解释参数变化为什么会改变图形外观
