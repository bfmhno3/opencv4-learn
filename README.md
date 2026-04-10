# opencv4-learn

一个按第一性原理组织的 OpenCV4 学习仓库，覆盖 **Python** 与 **C++** 两条路线，并且让两种语言在同一阶段下对照学习。

## 学习目标

这个仓库不把 OpenCV 当成“函数表”来记，而是按视觉问题的本质来学习：

1. 图像是什么：像素矩阵、通道、数据类型、坐标系统。
2. 图像能怎么变：读写、显示、绘制、几何变换、颜色变换、滤波。
3. 怎么从图像里提取结构：边缘、阈值、轮廓、形态学。
4. 怎么把基础操作串成完整流程：视频处理、经典检测、人脸与小项目。
5. 怎么在 **Python** 和 **C++** 中建立一一对应的实现能力。

## 仓库结构

```text
00_foundations/
01_io_and_display/
02_pixels_and_drawing/
03_color_and_intensity/
04_filtering_and_morphology/
05_geometric_transforms/
06_masking_thresholding_and_contours/
07_classical_object_analysis/
08_face_and_classical_recognition/
09_video_pipelines_and_mini_projects/
resources/
```

每个阶段目录都以 `README.md` 作为入口，并按学习进度逐步补齐：

- `python/`：Python 代码与实验
- `cpp/`：C++ 代码与实验
- 若干 `*.md`：实验说明、步骤、预期效果、扩展练习

`00_foundations` 是预备阶段，重点是心智模型和环境认知；`01` 之后逐步进入可执行实验。

## 推荐学习顺序

### 00_foundations
先建立图像与 OpenCV 的底层心智模型。

### 01_io_and_display
学会稳定地读图、读视频、显示结果、排查 I/O 问题。

### 02_pixels_and_drawing
把图像看成数组，并掌握最基础的绘制与像素级操作。

### 03_color_and_intensity
理解灰度、颜色空间、通道、直方图和亮度分布。

### 04_filtering_and_morphology
理解卷积核、模糊、边缘、膨胀、腐蚀与噪声处理。

### 05_geometric_transforms
掌握缩放、裁剪、平移、旋转、翻转及插值。

### 06_masking_thresholding_and_contours
学习目标区域提取、二值化、位操作与轮廓分析。

### 07_classical_object_analysis
把前面的处理步骤组合成经典视觉分析流程。

### 08_face_and_classical_recognition
使用现有的人脸资源完成经典人脸检测与识别实验。

### 09_video_pipelines_and_mini_projects
在视频与实时流中串起完整视觉管线，并做综合项目。

## Python 与 C++ 双轨学习方法

每个阶段都按下面顺序推进：

1. 先看阶段 `README.md`
2. 先做 `python/` 中的实验，快速建立直觉
3. 再做同阶段 `cpp/` 中的对应实验，理解 `cv::Mat`、编译、内存与 API 差异
4. 最后回到阶段文档做总结与扩展练习

### 对照时重点关注

- `numpy.ndarray` vs `cv::Mat`
- Python 动态脚本执行 vs C++ 编译与链接
- 颜色通道顺序与数据类型
- OpenCV 在两种语言中的 API 一致性与细节差异

## 资源现状与补充建议

当前仓库已经具备：

- `resources/photos/`：基础图片实验素材
- `resources/videos/`：视频实验素材
- `resources/faces/`：经典人脸相关素材

建议后续继续补充：

1. 低光、逆光、复杂背景图片
2. 纹理明显的图片（砖墙、树叶、道路、金属）
3. 含规则几何图形的测试图片
4. 至少 2~3 段不同场景的短视频
5. `resources/expected_outputs/`，用于保存标准结果截图

## 环境建议

### Python

仓库当前使用：

- `opencv-contrib-python`
- `numpy`
- `matplotlib`
- `caer`

如果需要新增 Python 包，统一使用：

```bash
uv add <package>
```

运行脚本使用：

```bash
uv run python path/to/script.py
```

### C++

建议在 Windows 上使用：

- CMake
- OpenCV 4.x
- 支持 C++17 的编译器

后续每个阶段的 `cpp/README.md` 或实验文档中会补充对应构建说明。

## 官方参考资料

- OpenCV Tutorials: <https://docs.opencv.org/4.x/>
- OpenCV-Python Tutorials: <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html>
- OpenCV C++ Tutorials: <https://docs.opencv.org/4.x/d9/df8/tutorial_root.html>

## 学习完成标准

完成这套路线后，你应该能够：

- 独立解释常见图像处理操作背后的数学与直觉
- 在 Python 中快速实现图像/视频处理实验
- 在 C++ 中复现相同实验并完成构建运行
- 组合多个 OpenCV 操作完成一个小型经典视觉项目
- 读懂官方教程、示例代码和常见 OpenCV 项目
