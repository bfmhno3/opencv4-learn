# 09 Video Pipelines and Mini Projects

## 阶段目标

把图像处理能力推广到连续帧，理解实时视觉系统中的处理顺序、效率、稳定性和可解释性，并最终完成一个小型综合项目。

## Python 实验

- `python/realtime_edges.py`（你来实现）
- `python/realtime_contours.py`（你来实现）
- `python/mini_project_live_pipeline.py`（你来实现）

## C++ 实验

- `cpp/realtime_edges.cpp`（你来实现）
- `cpp/realtime_contours.cpp`（你来实现）
- `cpp/mini_project_live_pipeline.cpp`（你来实现）

## 文档

- `video_pipeline_design.md`
- `mini_project_ideas.md`
- `resource_gap_checklist.md`
- `experiment_realtime_edges.md`
- `experiment_realtime_contours.md`
- `experiment_mini_project_live_pipeline.md`

## 建议实现顺序

1. 先复习 `01_io_and_display` 中的逐帧读取逻辑
2. 把单帧边缘或轮廓处理迁移到视频循环里
3. 观察分辨率、噪声、阈值对连续帧稳定性的影响
4. 加入实时显示、按键退出和结果标注
5. 组合成一个目标明确的小型实时项目
6. 先完成 Python，再用 C++ 复现并比较运行体验

## 阶段完成标准

你应该能：

- 解释视频处理和静态图像处理在数据流上的区别
- 设计一个清晰的 frame loop：读取、处理、显示、退出
- 说明输入、处理链路、输出和性能瓶颈
- 完成一个可运行、可解释的小型视频处理项目
