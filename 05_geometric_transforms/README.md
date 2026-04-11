# 05 Geometric Transforms

## 阶段目标

理解坐标变换与像素重采样：当图像被缩放、平移、旋转时，像素如何重新定位。

## Python 实验

- `python/resize_image.py`
- `python/translate_image.py`
- `python/rotate_image.py`
- `python/flip_image.py`

## C++ 实验

- `cpp/resize_image.cpp`（你来实现）
- `cpp/translate_image.cpp`（你来实现）
- `cpp/rotate_image.cpp`（你来实现）
- `cpp/flip_image.cpp`（你来实现）

## 文档

- `experiment_resize_and_interpolation.md`
- `experiment_translate_rotate_flip.md`

## 建议实现顺序

1. 先做 resize，理解尺寸变化和插值的关系
2. 再做平移、旋转、翻转，理解坐标如何重新映射
3. 观察旋转黑边、裁切和边界区域的变化
4. 最后对照 Python 与 C++，确认同一变换在两种语言中的参数含义

## 阶段完成标准

你应该能：

- 区分 resize 与 rescale
- 理解插值方式对效果的影响
- 理解旋转后的黑边与边界处理问题
