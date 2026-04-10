# 08 Face and Classical Recognition

## 阶段目标

基于仓库现有 `resources/faces/` 数据，学习经典人脸检测与经典人脸识别方法。

## Python 实验

- `python/face_detection_with_haar.py`（待补充）
- `python/face_recognition_with_lbph.py`（待补充）

## C++ 实验

- `cpp/face_detection_with_haar.cpp`（待补充）
- `cpp/face_recognition_with_lbph.cpp`（待补充）

## 推荐主题

- Haar Cascade 人脸检测
- 数据组织方式
- 训练/验证的基本概念
- LBPH 等经典识别方法的定位与局限

## 文档

- `face_dataset_notes.md`
- `classical_face_pipeline.md`
- `experiment_face_detection.md`
- `experiment_face_recognition.md`

## 建议实现顺序

1. 理解数据集目录结构
2. 完成人脸检测
3. 提取并准备训练样本
4. 训练经典识别器
5. 在验证集上观察结果与误差

## 阶段完成标准

你应该能理解：

- 为什么人脸任务需要数据集划分
- 为什么经典方法适合作为教学样例，而不是现代最强方案
- 如何把图像预处理、检测、识别串成完整流程
