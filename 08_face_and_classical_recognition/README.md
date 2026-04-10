# 08 Face and Classical Recognition

## 阶段目标

基于仓库现有 `resources/faces/` 数据，学习经典人脸检测与经典人脸识别方法，并理解“检测”和“识别”是两个不同问题。

## Python 实验

- `python/face_detection_with_haar.py`（你来实现）
- `python/face_recognition_with_lbph.py`（你来实现）

## C++ 实验

- `cpp/face_detection_with_haar.cpp`（你来实现）
- `cpp/face_recognition_with_lbph.cpp`（你来实现）

## 推荐主题

- Haar Cascade 人脸检测
- 数据组织方式与标签映射
- 训练 / 验证的基本概念
- LBPH 等经典识别方法的定位与局限
- Python 快速验证与 C++ 工程复现的对照关系

## 文档

- `face_dataset_notes.md`
- `classical_face_pipeline.md`
- `experiment_face_detection.md`
- `experiment_face_recognition.md`

## 建议实现顺序

1. 先理解 `resources/faces/train/` 和 `resources/faces/val/` 的目录结构
2. 先做人脸检测，理解检测参数、误检和漏检
3. 再直接基于现有 `train/` 与 `val/` 数据做人脸识别基线实验
4. 等识别基线跑通后，再尝试“检测 -> ROI 裁剪 -> 识别”的完整串联流程
5. 先在 Python 中完成最短可运行版本
6. 再用 C++ 复现同样的数据流和参数逻辑
7. 最后分析误检、漏检和识别错误的原因

## 阶段完成标准

你应该能理解：

- 为什么人脸任务必须区分检测与识别
- 为什么经典方法适合作为教学样例，而不是现代最强方案
- 为什么训练集与验证集必须分开
- 如何把预处理、检测、ROI 提取、训练、预测串成一个完整流程
