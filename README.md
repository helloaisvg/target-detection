# YOLOv5 目标检测项目

## 项目简介

本项目实现了基于YOLOv5的目标检测算法，能够在个人电脑上完成模型训练和目标检测任务。

## 项目结构

```
yolov5-5.0/
├── data/              # 数据集配置
├── models/            # 模型定义
├── utils/             # 工具函数
├── weights/           # 模型权重
├── runs/              # 训练和检测结果
├── train.py           # 训练脚本
├── detect.py          # 检测脚本
└── 目标检测实验报告.md # 实验报告
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch torchvision numpy opencv-python Pillow PyYAML
```

### 2. 训练模型

```bash
python train.py --img 640 --batch 8 --epochs 200 \
    --data ./data/coco128.yaml \
    --weights ./weights/yolov5s.pt \
    --device cpu
```

### 3. 运行检测

```bash
# 检测单张图片
python detect.py --source <图片路径> \
    --weights ./weights/coco128.pt \
    --conf 0.15 --device cpu

# 检测整个目录
python detect.py --source <目录路径> \
    --weights ./weights/coco128.pt \
    --conf 0.15 --device cpu
```

## 实验结果

- **训练**: 200个epoch，最终loss: 0.0003
- **检测**: 128张图片，34张成功检测（26.6%）
- **模型**: weights/coco128.pt (28MB)

## 详细文档

请查看 `目标检测实验报告.md` 获取完整的实验报告。

## 注意事项

- 本项目在CPU模式下运行，检测速度较慢
- VM虚拟机运行，GPU直通出现问题故只用CPU
- 训练数据量较小，模型效果有限
