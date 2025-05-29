# RETFound - MindSpore版本

这是RETFound项目的MindSpore框架实现版本，从原始的PyTorch版本转换而来。

## 主要变化

### 框架转换
- **PyTorch → MindSpore**: 将所有PyTorch代码转换为MindSpore实现
- **模型架构**: Vision Transformer模型完全用MindSpore重新实现
- **训练流程**: 使用MindSpore的训练API和优化器
- **数据处理**: 使用MindSpore的数据集API和变换操作

### 文件结构

```
├── models_vit_mindspore.py          # MindSpore版本的ViT模型
├── main_finetune_mindspore.py       # MindSpore版本的微调脚本
├── engine_finetune_mindspore.py     # MindSpore版本的训练引擎
├── util_mindspore/                  # MindSpore版本的工具模块
│   ├── misc.py                      # 杂项工具函数
│   ├── datasets.py                  # 数据集处理
│   ├── lr_sched.py                  # 学习率调度
│   └── pos_embed.py                 # 位置编码
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

### 性能优化
1. **图模式**: 默认使用GRAPH_MODE进行性能优化
2. **JIT编译**: 支持@ms.jit装饰器
3. **内存优化**: 自动内存管理和优化

## 注意事项

1. **预训练权重**: 需要将PyTorch的预训练权重转换为MindSpore格式
2. **数据格式**: 确保数据集格式与MindSpore兼容
3. **设备选择**: 根据硬件环境选择CPU或GPU
4. **版本兼容**: 建议使用MindSpore 2.2.0或更高版本

## 故障排除

### 常见问题
1. **内存不足**: 减小batch_size或使用梯度累积
2. **CUDA错误**: 检查MindSpore GPU版本与CUDA版本匹配
3. **数据加载慢**: 调整num_workers参数

### 性能调优
1. **使用图模式**: 确保设置`mode=context.GRAPH_MODE`
2. **数据预处理**: 使用MindSpore的数据预处理操作
3. **混合精度**: 启用自动混合精度训练

## 贡献

欢迎提交Issue和Pull Request来改进MindSpore版本的实现。

## 许可证

与原项目保持一致的许可证。 