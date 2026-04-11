# 06 Masking, Thresholding and Contours

## 阶段目标

学习如何从整张图里提取“我真正关心的那一部分”，并理解二值化为何是轮廓分析的重要前处理。

## Python 实验

- `python/image_masking.py`
- `python/bitwise_operation.py`
- `python/contour_detection.py`
- `python/thresholding_demo.py`

## C++ 实验

- `cpp/image_masking.cpp`（你来实现）
- `cpp/bitwise_operation.cpp`（你来实现）
- `cpp/thresholding_demo.cpp`（你来实现）
- `cpp/contour_detection.cpp`（你来实现）

## 文档

- `experiment_masking_and_bitwise.md`
- `experiment_thresholding.md`
- `experiment_contour_detection.md`

## 建议实现顺序

1. 先做 mask 和 bitwise 操作，理解如何限制分析区域
2. 再做阈值化，观察连续强度如何变成二值前景 / 背景
3. 然后在二值图或边缘图上提取轮廓
4. 最后分析前处理变化为什么会直接影响轮廓结果，并用 C++ 复现同样流程

## 阶段完成标准

你应该能：

- 使用 mask 限制分析区域
- 理解 bitwise and/or/xor/not 的效果
- 解释二值化在目标区域提取中的作用
- 解释轮廓为何依赖前处理质量
