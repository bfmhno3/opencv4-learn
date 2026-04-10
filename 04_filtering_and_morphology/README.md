# 04 Filtering and Morphology

## 阶段目标

理解局部邻域运算：一个像素的新值，往往取决于它周围的一小块区域。

## Python 实验

- `python/gaussian_blur.py`
- `python/blurring.py`
- `python/canny_edge_detector.py`
- `python/dilation.py`
- `python/erosion.py`

## C++ 实验

- `cpp/gaussian_blur.cpp`（你来实现）
- `cpp/blurring.cpp`（你来实现）
- `cpp/canny_edge_detector.cpp`（你来实现）
- `cpp/dilation.cpp`（你来实现）
- `cpp/erosion.cpp`（你来实现）

## 文档

- `experiment_blur_and_noise.md`
- `experiment_canny_edges.md`
- `experiment_morphology.md`

## 阶段完成标准

你应该能：

- 理解平均、高斯、中值、双边滤波的区别
- 理解为什么 Canny 前常常先做模糊
- 理解膨胀、腐蚀在连接边缘和消除小噪声中的作用
