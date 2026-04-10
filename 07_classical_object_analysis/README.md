# 07 Classical Object Analysis

## 阶段目标

把 06 阶段学到的阈值化、边缘提取、轮廓检测、形态学等基础操作串起来，形成一个完整、可解释、可视化的经典视觉分析流程。

## Python 实验

- `python/shape_analysis_pipeline.py`（你来实现）
- `python/contour_filtering_pipeline.py`（你来实现）

## C++ 实验

- `cpp/shape_analysis_pipeline.cpp`（你来实现）
- `cpp/contour_filtering_pipeline.cpp`（你来实现）

## 文档

- `pipeline_design.md`
- `shape_analysis_plan.md`
- `experiment_shape_analysis_pipeline.md`
- `experiment_contour_filtering_pipeline.md`

## 建议实现顺序

1. 从 `resources/photos/` 里挑选边界清晰、背景相对简单的图片
2. 完成灰度化、模糊、阈值化或 Canny 预处理
3. 提取轮廓并观察误检、漏检情况
4. 计算面积、周长、包围框、长宽比等描述量
5. 用规则过滤目标轮廓并完成可视化标注
6. 先完成 Python 版本，再用 C++ 复现相同流程

## 阶段完成标准

你应该能：

- 解释为什么预处理质量会直接影响轮廓质量
- 把多个基础 OpenCV 操作串成一个完整分析流程
- 用规则而不是模型完成一个小型目标分析任务
- 对照 Python / C++ 两个版本说明同一条数据流是如何实现的
