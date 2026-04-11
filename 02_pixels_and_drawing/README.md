# 02 Pixels and Drawing

## 阶段目标

通过像素访问、切片和绘图，把图像真正理解成“可操作的数据结构”。

## Python 实验

- `python/draw_line.py`
- `python/draw_rectangle.py`
- `python/draw_circle.py`
- `python/add_text.py`
- `python/cropping_image.py`

## C++ 实验

- `cpp/draw_line.cpp`（你来实现）
- `cpp/draw_rectangle.cpp`（你来实现）
- `cpp/draw_circle.cpp`（你来实现）
- `cpp/add_text.cpp`（你来实现）
- `cpp/cropping_image.cpp`（你来实现）

## 文档

- `experiment_canvas_and_coordinates.md`
- `experiment_drawing_shapes.md`
- `experiment_text_and_crop.md`

## 建议实现顺序

1. 先用空白画布理解坐标原点、宽高和颜色填充
2. 再分别绘制直线、矩形、圆和文本
3. 然后切换到真实图片，练习 ROI 裁剪与局部修改
4. 最后对照 Python 和 C++，确认坐标与参数含义保持一致

## 阶段完成标准

你应该能：

- 理解图像坐标原点与方向
- 解释 thickness、filled、ROI 的意义
- 在 Python 与 C++ 中绘制和裁剪图像
