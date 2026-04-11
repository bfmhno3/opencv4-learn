# 01 IO and Display

## 阶段目标

先掌握输入与输出，因为视觉实验的第一步永远是：正确获取数据、正确展示结果。

## 核心问题

- 如何稳定读取图片？
- 如何逐帧读取视频或摄像头？
- 如何发现读图失败、视频流结束、窗口卡住等问题？
- Python 与 C++ 的 I/O 接口差异在哪里？

## Python 实验

- `python/read_image.py`
- `python/read_videos.py`
- `python/rescale_video.py`

## C++ 实验

- `cpp/read_image.cpp`（你来实现）
- `cpp/read_videos.cpp`（你来实现）
- `cpp/rescale_video.cpp`（你来实现）

## 文档

- `experiment_read_image.md`
- `experiment_read_video.md`
- `experiment_rescale_video.md`

## 建议实现顺序

1. 先用单张图片跑通 `imread` 和 `imshow`
2. 再用视频文件理解逐帧读取、空帧和退出条件
3. 然后加入缩放逻辑，比较不同尺寸对显示与速度的影响
4. 先完成 Python 版本，再用 C++ 复现同样的 I/O 流程

## 阶段完成标准

你应该能：

- 解释 `imread` / `VideoCapture` 的基本工作方式
- 判断空图像、空帧的常见原因
- 使用 Python 和 C++ 跑通最基础的图像/视频读写流程
