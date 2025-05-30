# RETFound PyTorch到PaddlePaddle转换说明

## 概述

本文档详细记录了将RETFound（视网膜图像基础模型）项目从PyTorch框架转换为PaddlePaddle框架的完整过程。RETFound是一个基于MAE（Masked Autoencoder）的视网膜图像疾病检测基础模型。

## 转换文件列表

### 核心文件转换

| 原文件 | 状态 | 主要变化 |
|--------|------|----------|
| `requirements.txt` | ✅ 已转换 | 替换PyTorch依赖为PaddlePaddle |
| `models_vit.py` | ✅ 已转换 | 完全重写Vision Transformer架构 |
| `engine_finetune.py` | ✅ 已转换 | 训练和评估引擎转换 |
| `main_finetune.py` | ✅ 已转换 | 主训练脚本转换 |
| `util/datasets.py` | ✅ 已转换 | 数据处理模块转换 |
| `util/misc.py` | ✅ 已转换 | 工具函数转换 |
| `util/lr_sched.py` | ✅ 已转换 | 学习率调度器转换 |
| `util/lr_decay.py` | ✅ 已转换 | 学习率衰减转换 |
| `util/pos_embed.py` | ✅ 已转换 | 位置编码转换 |

## 主要API转换对照表

### 1. 基础导入和框架

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `import torch` | `import paddle` | 主框架导入 |
| `import torch.nn as nn` | `import paddle.nn as nn` | 神经网络模块 |
| `import torch.nn.functional as F` | `import paddle.nn.functional as F` | 函数式API |
| `torch.utils.tensorboard` | `visualdl` | 可视化工具 |
| `timm.models` | 自定义实现 | 模型库替换 |

### 2. 模型定义

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `torch.nn.Module` | `paddle.nn.Layer` | 基础模型类 |
| `torch.nn.Linear` | `paddle.nn.Linear` | 线性层 |
| `torch.nn.Conv2d` | `paddle.nn.Conv2D` | 卷积层 |
| `torch.nn.LayerNorm` | `paddle.nn.LayerNorm` | 归一化层 |
| `model.parameters()` | `model.parameters()` | 参数获取（相同） |

### 3. 张量操作

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `torch.tensor()` | `paddle.to_tensor()` | 张量创建 |
| `tensor.to(device)` | 自动设备管理 | 设备分配 |
| `torch.cat()` | `paddle.concat()` | 张量拼接 |
| `tensor.permute()` | `tensor.transpose()` | 维度转换 |
| `tensor.view()` | `tensor.reshape()` | 形状变换 |
| `tensor.expand()` | `tensor.expand()` | 形状扩展（相同） |

### 4. 训练相关

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `optimizer.zero_grad()` | `optimizer.clear_grad()` | 梯度清零 |
| `loss.backward()` | `loss.backward()` | 反向传播（相同） |
| `optimizer.step()` | `optimizer.step()` | 参数更新（相同） |
| `model.train()` | `model.train()` | 训练模式（相同） |
| `model.eval()` | `model.eval()` | 评估模式（相同） |

### 5. 数据加载

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `torch.utils.data.DataLoader` | `paddle.io.DataLoader` | 数据加载器 |
| `torch.utils.data.Dataset` | `paddle.io.Dataset` | 数据集基类 |
| `torchvision.datasets` | `paddle.vision.datasets` | 内置数据集 |
| `torchvision.transforms` | `paddle.vision.transforms` | 数据变换 |

### 6. 分布式训练

| PyTorch | PaddlePaddle | 说明 |
|---------|--------------|------|
| `torch.distributed` | `paddle.distributed` | 分布式模块 |
| `DistributedDataParallel` | `DataParallel` | 并行训练 |
| `DistributedSampler` | `DistributedSampler` | 分布式采样（相同API） |

## 关键转换要点

### 1. Vision Transformer架构转换

- **完全重写**：由于PaddlePaddle没有直接对应的timm库，完全重新实现了Vision Transformer架构
- **组件包括**：
  - `PatchEmbed`: 图像块嵌入层
  - `Attention`: 多头自注意力机制
  - `Mlp`: 多层感知器
  - `Block`: Transformer块
  - `DropPath`: 随机深度
  - `VisionTransformer`: 完整的ViT模型

### 2. 数据增强转换

- **Mixup实现**：自定义实现Mixup数据增强（原来使用timm.data.Mixup）
- **随机擦除**：自定义实现RandomErasing
- **基础变换**：使用paddle.vision.transforms替代torchvision.transforms

### 3. 训练引擎改造

- **自动混合精度**：`torch.cuda.amp` → `paddle.amp`
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_` → `paddle.nn.utils.clip_grad_norm_`
- **指标计算**：保留sklearn的指标计算，添加自定义accuracy函数

### 4. 分布式支持

- **初始化方式**：`torch.distributed.init_process_group` → `paddle.distributed.init_parallel_env`
- **同步操作**：保持相似的API接口
- **设备管理**：`torch.cuda.set_device` → `paddle.device.cuda.set_device`

## 依赖变化

### 移除的依赖
```text
- torch
- torchvision
- timm
- tensorboard
```

### 新增的依赖
```text
+ paddlepaddle-gpu>=2.6.0
+ paddlenlp>=2.7.0
+ visualdl
```

### 保留的依赖
```text
- opencv-python
- Pillow
- numpy
- matplotlib
- scikit-learn
- scikit-multilearn
- huggingface-hub
- pycm
```

## 使用说明

### 1. 环境安装

```bash
# 创建conda环境
conda create -n retfound-paddle python=3.8 -y
conda activate retfound-paddle

# 安装PaddlePaddle
pip install paddlepaddle-gpu>=2.6.0

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 模型训练

```bash
# 使用PaddlePaddle版本进行训练
python main_finetune.py \
    --model RETFound_mae \
    --finetune RETFound_mae_natureCFP \
    --batch_size 16 \
    --epochs 100 \
    --data_path ./data \
    --output_dir ./output \
    --task retfound_test
```

### 3. 模型评估

```bash
# 评估模式
python main_finetune.py \
    --model RETFound_mae \
    --eval \
    --resume ./output/checkpoint-best.pth \
    --data_path ./data
```

## 注意事项

### 1. API差异
- **参数名称**：某些API的参数名略有不同（如`eps` vs `epsilon`）
- **默认值**：部分函数的默认参数值可能不同
- **形状操作**：维度转换方法有所不同（`permute` vs `transpose`）

### 2. 性能考虑
- **内存使用**：PaddlePaddle的内存管理策略与PyTorch不同
- **计算图**：动态图机制可能在某些操作上有性能差异
- **优化器**：学习率调度策略需要适配PaddlePaddle的接口

### 3. 模型兼容性
- **预训练权重**：需要转换PyTorch权重格式到PaddlePaddle格式
- **模型保存**：使用`paddle.save`和`paddle.load`替代torch对应方法
- **状态字典**：使用`set_state_dict`替代`load_state_dict`

### 4. 调试建议
- **日志输出**：使用VisualDL替代TensorBoard进行可视化
- **错误处理**：注意PaddlePaddle特有的错误信息和调试方法
- **版本兼容**：确保PaddlePaddle版本与其他依赖的兼容性