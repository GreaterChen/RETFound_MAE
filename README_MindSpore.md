# RETFound - MindSpore版本

这是RETFound项目的MindSpore框架实现版本，从原始的PyTorch版本转换而来。

## 主要变化

### 框架转换
- **PyTorch → MindSpore**: 将所有PyTorch代码转换为MindSpore实现
- **模型架构**: Vision Transformer模型完全用MindSpore重新实现
- **训练流程**: 使用MindSpore的训练API和优化器
- **数据处理**: 使用MindSpore的数据集API和变换操作
- **权重转换**: 自动从HuggingFace下载PyTorch权重并转换为MindSpore格式

### 文件结构

```
├── models_vit_mindspore.py          # MindSpore版本的ViT模型
├── main_finetune_mindspore.py       # MindSpore版本的微调脚本
├── engine_finetune_mindspore.py     # MindSpore版本的训练引擎
├── util_mindspore/                  # MindSpore版本的工具模块
│   ├── misc.py                      # 杂项工具函数
│   ├── datasets.py                  # 数据集处理
│   ├── lr_sched.py                  # 学习率调度
│   ├── pos_embed.py                 # 位置编码
│   └── weight_converter.py          # PyTorch权重转换工具
├── convert_weights_example.py       # 权重转换示例脚本
├── requirements_mindspore.txt       # MindSpore版本依赖
└── README_MindSpore.md             # 本文档
```

## 安装环境

1. 创建conda环境:
```bash
conda create -n retfound_mindspore python=3.9 -y
conda activate retfound_mindspore
```

2. 安装MindSpore:
```bash
# CPU版本
pip install mindspore

# GPU版本 (CUDA 11.6)
pip install mindspore-gpu

# 或者根据你的CUDA版本选择合适的安装命令
# 详见: https://www.mindspore.cn/install
```

3. 安装其他依赖:
```bash
pip install -r requirements_mindspore.txt
```

4. (可选) 安装PyTorch用于权重转换:
```bash
# 仅在需要从PyTorch权重转换时安装
pip install torch torchvision
```

## 权重转换功能

### 自动权重转换
本MindSpore版本支持自动从HuggingFace下载PyTorch预训练权重并转换为MindSpore格式：

1. **首次使用**: 自动下载PyTorch权重并转换为MindSpore格式，保存到本地缓存
2. **后续使用**: 直接加载本地缓存的MindSpore权重，无需重复转换
3. **智能缓存**: 根据模型名称自动管理权重缓存

### 权重转换示例
```bash
# 转换RETFound_mae_natureCFP权重
python convert_weights_example.py --finetune RETFound_mae_natureCFP

# 指定缓存目录
python convert_weights_example.py --finetune RETFound_mae_natureCFP --cache_dir ./my_weights_cache
```

## 使用方法

### 微调训练

```bash
python main_finetune_mindspore.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 16 \
    --epochs 100 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD \
    --input_size 224 \
    --task RETFound_mae_meh-IDRiD \
    --finetune RETFound_mae_meh \
    --weights_cache_dir ./weights_cache \
    --device GPU
```

### 仅评估

```bash
python main_finetune_mindspore.py \
    --model RETFound_mae \
    --eval \
    --global_pool \
    --batch_size 16 \
    --nb_classes 5 \
    --data_path ./IDRiD \
    --input_size 224 \
    --resume ./checkpoint-best.ckpt \
    --device GPU
```

## 主要特性

### 模型架构
- **Vision Transformer**: 完整的ViT实现，包括patch embedding、multi-head attention、MLP等
- **位置编码**: 支持2D正弦余弦位置编码和插值
- **全局池化**: 支持全局平均池化和CLS token两种方式

### 训练功能
- **混合精度训练**: 使用MindSpore的自动混合精度
- **数据增强**: 支持Mixup、CutMix、随机擦除等
- **学习率调度**: 余弦退火学习率调度
- **层级学习率衰减**: 支持不同层使用不同的学习率

### 权重管理
- **自动转换**: 从PyTorch权重自动转换为MindSpore格式
- **智能缓存**: 本地缓存转换后的权重，避免重复转换
- **兼容性检查**: 自动检查权重与模型的兼容性

### 评估指标
- **分类指标**: 准确率、F1分数、AUC、精确率、召回率
- **多类分类**: 支持宏平均和加权平均
- **混淆矩阵**: 详细的分类结果分析

## 与PyTorch版本的差异

### API差异
1. **模型定义**: `nn.Module` → `nn.Cell`
2. **前向传播**: `forward()` → `construct()`
3. **优化器**: `torch.optim` → `mindspore.nn`
4. **数据加载**: `torch.utils.data` → `mindspore.dataset`

### 训练流程差异
1. **梯度计算**: 使用`ms.value_and_grad()`
2. **设备管理**: 使用`context.set_context()`
3. **分布式训练**: 使用MindSpore的通信原语

### 权重处理差异
1. **权重格式**: 自动转换PyTorch权重格式到MindSpore
2. **权重加载**: 使用MindSpore的权重加载机制
3. **权重缓存**: 本地缓存转换后的权重

### 性能优化
1. **图模式**: 默认使用GRAPH_MODE进行性能优化
2. **JIT编译**: 支持@ms.jit装饰器
3. **内存优化**: 自动内存管理和优化

## 注意事项

### 权重转换
1. **首次使用**: 需要安装PyTorch和huggingface_hub进行权重转换
2. **网络连接**: 首次下载需要稳定的网络连接到HuggingFace
3. **存储空间**: 确保有足够空间存储转换后的权重文件

### 环境要求
1. **MindSpore版本**: 建议使用MindSpore 2.2.0或更高版本
2. **Python版本**: 支持Python 3.7-3.9
3. **设备支持**: 支持CPU和GPU训练

### 数据格式
1. **数据集结构**: 使用标准的ImageFolder格式
2. **图像格式**: 支持常见的图像格式(JPG, PNG等)
3. **标签格式**: 使用文件夹名作为类别标签

## 故障排除

### 常见问题
1. **权重转换失败**: 检查PyTorch和huggingface_hub是否正确安装
2. **内存不足**: 减小batch_size或使用梯度累积
3. **CUDA错误**: 检查MindSpore GPU版本与CUDA版本匹配
4. **网络连接**: 确保能够访问HuggingFace Hub

### 性能调优
1. **使用图模式**: 确保设置`mode=context.GRAPH_MODE`
2. **数据预处理**: 使用MindSpore的数据预处理操作
3. **混合精度**: 启用自动混合精度训练
4. **权重缓存**: 使用本地权重缓存避免重复转换

## 支持的预训练模型

| 模型名称 | 描述 | HuggingFace链接 |
|---------|------|----------------|
| RETFound_mae_natureCFP | Nature论文CFP模型 | [链接](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) |
| RETFound_mae_natureOCT | Nature论文OCT模型 | [链接](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) |
| RETFound_mae_meh | MEH数据集模型 | [链接](https://huggingface.co/YukunZhou/RETFound_mae_meh) |
| RETFound_mae_shanghai | 上海数据集模型 | [链接](https://huggingface.co/YukunZhou/RETFound_mae_shanghai) |
| RETFound_dinov2_meh | DINOV2 MEH模型 | [链接](https://huggingface.co/YukunZhou/RETFound_dinov2_meh) |
| RETFound_dinov2_shanghai | DINOV2上海模型 | [链接](https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai) |

## 贡献

欢迎提交Issue和Pull Request来改进MindSpore版本的实现。

## 许可证

与原项目保持一致的许可证。 