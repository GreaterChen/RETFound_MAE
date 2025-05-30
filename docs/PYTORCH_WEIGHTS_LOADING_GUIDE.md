# RETFound PyTorch权重加载到PaddlePaddle指南

## 概述

本指南详细说明了如何将RETFound项目的PyTorch预训练权重转换并加载到PaddlePaddle模型中。我们提供了完整的转换工具和示例脚本，确保与原始PyTorch模型的完全兼容性。

## 关键特性

✅ **完全兼容** - 模型架构与PyTorch版本完全一致  
✅ **自动转换** - 内置权重转换工具，自动处理维度差异  
✅ **HuggingFace支持** - 直接从HuggingFace Hub下载和转换  
✅ **镜像站支持** - 支持HF镜像站，解决国内访问问题  
✅ **便捷接口** - 一行代码完成权重加载  
✅ **验证工具** - 内置模型推理测试

## 快速开始

### 1. 最简单的使用方法

```python
import paddle
from models_vit import RETFound_mae

# 创建模型
model = RETFound_mae(num_classes=1000)

# 一行代码加载PyTorch权重（自动转换）
model.load_pytorch_weights("path/to/pytorch_retfound_mae.pth")

# 模型即可使用
model.eval()
with paddle.no_grad():
    output = model(paddle.randn([1, 3, 224, 224]))
```

### 2. 使用示例脚本

```bash
# 使用本地PyTorch权重文件
python convert_ckpt.py --use_local --local_model_path /path/to/model.pth --test_inference

# 从HuggingFace Hub下载并转换
python convert_ckpt.py --use_huggingface --model_name YukunZhou/RETFound_mae_meh --hf_endpoint https://hf-mirror.com

# 使用HF Token访问私有仓库
python convert_ckpt.py --use_huggingface --hf_token your_token_here --model_name private/model
```

## 支持的模型

| 模型名称 | HuggingFace Hub | 架构参数 | 描述 |
|---------|----------------|----------|------|
| RETFound_MAE | `YukunZhou/RETFound_mae_meh` | ViT-Large, 1024d, 24层, patch=16 | MAE预训练的RETFound模型 |
| RETFound_DINOv2 | `ycxia/RETFound_cf_dinov2` | ViT-Large, 1024d, 24层, patch=14 | DINOv2预训练的RETFound模型 |

## 核心功能

### 1. 自动权重转换 (`model.load_pytorch_weights()`)

```python
from models_vit import RETFound_mae

model = RETFound_mae(num_classes=5)  # 根据需要设置类别数

# 方法会自动完成以下步骤：
# 1. 加载PyTorch权重文件
# 2. 转换权重格式（处理维度差异）
# 3. 保存为PaddlePaddle格式
# 4. 加载到当前模型
success = model.load_pytorch_weights("/path/to/pytorch_model.pth")
```

### 2. 完整转换脚本 (`convert_ckpt.py`)

#### 2.1 本地文件模式

```bash
# 基本用法
python convert_ckpt.py \
    --use_local \
    --local_model_path "/home/chenlb24/RETFound_MAE/RETFound_mae_meh.pth" \
    --test_inference

# 保存PaddlePaddle格式模型
python convert_ckpt.py \
    --use_local \
    --local_model_path "/path/to/model.pth" \
    --save_paddle_model \
    --paddle_model_path "my_retfound_paddle.pdparams"
```

#### 2.2 HuggingFace下载模式

```bash
# 使用默认镜像站
python convert_ckpt.py \
    --use_huggingface \
    --model_name "YukunZhou/RETFound_mae_meh" \
    --hf_endpoint "https://hf-mirror.com"

# 使用Token访问私有仓库
python convert_ckpt.py \
    --use_huggingface \
    --model_name "private/retfound_model" \
    --hf_token "your_hf_token_here" \
    --cache_dir "./models"
```

### 3. 权重转换工具 (`util/weight_converter.py`)

#### 3.1 直接转换权重文件

```python
from util.weight_converter import convert_retfound_weights

# 转换权重文件
success = convert_retfound_weights(
    pytorch_model_path="retfound_mae.pth",
    paddle_model_path="retfound_mae_paddle.pdparams"
)
```

#### 3.2 加载转换后的权重

```python
from util.weight_converter import load_converted_weights_to_model

# 加载已转换的权重到模型
success = load_converted_weights_to_model(
    model=model,
    paddle_model_path="retfound_mae_paddle.pdparams",
    strict=False  # 允许部分权重不匹配
)
```

## 模型架构详情

### PaddlePaddle实现与PyTorch的一致性

我们的实现确保了以下关键组件与PyTorch完全一致：

#### 1. Vision Transformer结构

```python
VisionTransformer(
    ├── patch_embed (PatchEmbed)          # 图像分块嵌入
    │   └── proj (Conv2D)                 # 16x16或14x14卷积
    ├── cls_token (Parameter)             # 分类令牌
    ├── pos_embed (Parameter)             # 位置编码
    ├── blocks (LayerList)                # Transformer块序列
    │   ├── norm1 (LayerNorm)             # 注意力前归一化
    │   ├── attn (Attention)              # 多头自注意力
    │   │   ├── qkv (Linear)              # Q、K、V投影
    │   │   └── proj (Linear)             # 输出投影
    │   ├── norm2 (LayerNorm)             # MLP前归一化
    │   └── mlp (Mlp)                     # 前馈网络
    │       ├── fc1 (Linear)              # 第一个线性层
    │       └── fc2 (Linear)              # 第二个线性层
    ├── norm (LayerNorm)                  # 最终归一化
    └── head (Linear)                     # 分类头
)
```

#### 2. RETFound模型配置

```python
# RETFound MAE
RETFound_mae = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=1024,    # ViT-Large配置
    depth=24,          # 24层Transformer
    num_heads=16,      # 16个注意力头
    mlp_ratio=4,       # MLP隐藏层倍数
    qkv_bias=True,     # QKV偏置
    num_classes=1000   # 可调整
)

# RETFound DINOv2  
RETFound_dinov2 = VisionTransformer(
    img_size=224,
    patch_size=14,     # DINOv2使用14x14 patch
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    num_classes=1000
)
```

## 权重转换技术细节

### 1. 自动处理的转换

我们的转换工具自动处理以下差异：

#### 1.1 线性层权重转置

```python
# PyTorch Linear层权重形状: [out_features, in_features]
# PaddlePaddle Linear层权重形状: [in_features, out_features]

# 自动转置的权重类型
transpose_weights = [
    'qkv.weight',      # 注意力QKV投影
    'proj.weight',     # 注意力输出投影
    'fc1.weight',      # MLP第一层
    'fc2.weight',      # MLP第二层
    'head.weight'      # 分类头
]
```

#### 1.2 权重名称映射

```python
# 权重名称保持一致，无需特殊映射
pytorch_name -> paddle_name
"patch_embed.proj.weight" -> "patch_embed.proj.weight"
"blocks.0.attn.qkv.weight" -> "blocks.0.attn.qkv.weight"
"cls_token" -> "cls_token"
"pos_embed" -> "pos_embed"
```

#### 1.3 数据类型和设备处理

```python
# 自动转换PyTorch tensor为PaddlePaddle tensor
pytorch_tensor = torch.load(...)
paddle_tensor = paddle.to_tensor(pytorch_tensor.detach().cpu().numpy())
```

### 2. 转换过程

```python
class PyTorchToPaddleConverter:
    def convert_state_dict(self, pytorch_state_dict):
        paddle_state_dict = OrderedDict()
        
        for pytorch_name, pytorch_weight in pytorch_state_dict.items():
            # 1. 转换权重名称（通常保持不变）
            paddle_name = self.convert_weight_name(pytorch_name)
            
            # 2. 转换权重值（处理转置等）
            paddle_weight = self.convert_weight_value(pytorch_name, pytorch_weight)
            
            paddle_state_dict[paddle_name] = paddle_weight
            
        return paddle_state_dict
```

## 使用示例

### 1. 基本使用

```python
import paddle
from models_vit import RETFound_mae

# 创建模型
model = RETFound_mae(num_classes=5)  # 5分类任务

# 加载预训练权重
model.load_pytorch_weights("/path/to/RETFound_mae_meh.pth")

# 使用模型
model.eval()
input_tensor = paddle.randn([1, 3, 224, 224])

with paddle.no_grad():
    # 分类预测
    logits = model(input_tensor)
    print(f"分类输出: {logits.shape}")  # [1, 5]
    
    # 特征提取
    features = model.forward_features(input_tensor)
    print(f"特征输出: {features.shape}")  # [1, 1024]
```

### 2. 完整训练流程

```python
import paddle
import paddle.optimizer as optim
from models_vit import RETFound_mae

# 1. 创建模型并加载预训练权重
model = RETFound_mae(num_classes=5)
model.load_pytorch_weights("/path/to/RETFound_mae_meh.pth")

# 2. 冻结部分层（可选）
for param in model.patch_embed.parameters():
    param.stop_gradient = True

# 3. 设置优化器
optimizer = optim.AdamW(
    parameters=model.parameters(),
    learning_rate=1e-4,
    weight_decay=0.05
)

# 4. 训练
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.clear_grad()
    
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    loss.backward()
    optimizer.step()
```

### 3. 特征提取

```python
# 使用RETFound作为特征提取器
class RETFoundFeatureExtractor(paddle.nn.Layer):
    def __init__(self, pretrained_path):
        super().__init__()
        self.backbone = RETFound_mae(num_classes=1000)
        self.backbone.load_pytorch_weights(pretrained_path)
        
        # 移除分类头
        self.backbone.head = paddle.nn.Identity()
    
    def forward(self, x):
        return self.backbone.forward_features(x)

# 使用
feature_extractor = RETFoundFeatureExtractor("/path/to/weights.pth")
features = feature_extractor(images)  # [B, 1024]
```

## 脚本参数详解

### convert_ckpt.py 参数

#### HuggingFace相关
- `--hf_token`: HuggingFace访问令牌
- `--hf_endpoint`: HF镜像站地址（默认：https://hf-mirror.com）
- `--model_name`: HF模型仓库名称（默认：YukunZhou/RETFound_mae_meh）
- `--cache_dir`: 模型缓存目录（默认：./models）

#### 模式选择
- `--use_huggingface`: 从HuggingFace下载模式
- `--use_local`: 使用本地文件模式（默认）
- `--local_model_path`: 本地PyTorch模型文件路径

#### 功能选项
- `--test_inference`: 测试模型推理（默认：True）
- `--save_paddle_model`: 保存PaddlePaddle格式模型（默认：False）
- `--paddle_model_path`: PaddlePaddle模型保存路径
- `--max_retries`: 下载重试次数（默认：3）

#### 设备设置
- `--device`: 计算设备（auto/gpu/cpu，默认：auto）

## 故障排除

### 1. 权重加载失败

#### 问题：模型权重加载失败

```bash
❌ 权重加载失败
```

**解决方案：**

```python
# 1. 检查PyTorch权重文件格式
import torch
checkpoint = torch.load("model.pth", map_location='cpu')
print(f"权重文件键: {list(checkpoint.keys())}")

# 2. 检查权重结构
if 'model' in checkpoint:
    weights = checkpoint['model']
elif 'state_dict' in checkpoint:
    weights = checkpoint['state_dict']
else:
    weights = checkpoint
    
print(f"权重参数数量: {len(weights)}")
```

### 2. HuggingFace下载失败

#### 问题：网络连接或认证失败

```bash
❌ HuggingFace模型下载/加载失败
```

**解决方案：**

```bash
# 1. 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
python convert_ckpt.py --use_huggingface --hf_endpoint https://hf-mirror.com

# 2. 设置代理（如果需要）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 3. 检查网络连接
curl -I https://hf-mirror.com

# 4. 手动下载后使用本地模式
wget https://hf-mirror.com/YukunZhou/RETFound_mae_meh/pytorch_model.bin
python convert_ckpt.py --use_local --local_model_path pytorch_model.bin
```

### 3. 维度不匹配

#### 问题：模型参数维度不匹配

```bash
ValueError: Shape mismatch for parameter xxx
```

**解决方案：**

```python
# 1. 检查模型配置
model = RETFound_mae(
    num_classes=1000,  # 确保与预训练模型一致
    global_pool=False  # 检查池化方式
)

# 2. 查看权重形状
paddle_weights = paddle.load("converted_weights.pdparams")
model_state = model.state_dict()

for name in model_state.keys():
    if name in paddle_weights:
        model_shape = model_state[name].shape
        weight_shape = paddle_weights[name].shape
        if model_shape != weight_shape:
            print(f"不匹配: {name} - 模型:{model_shape}, 权重:{weight_shape}")
```

### 4. 内存不足

#### 问题：转换过程中内存不足

**解决方案：**

```python
import gc
import paddle

# 分步骤处理
def memory_efficient_conversion(pytorch_path, paddle_path):
    # 1. 转换权重
    success = convert_retfound_weights(pytorch_path, paddle_path)
    gc.collect()
    
    # 2. 加载到模型
    if success:
        model = RETFound_mae(num_classes=1000)
        load_converted_weights_to_model(model, paddle_path, strict=False)
        
        # 清理内存
        gc.collect()
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
```

## 性能优化

### 1. 批量转换

```python
# 批量转换多个模型
models_config = [
    {
        "pytorch_path": "retfound_mae.pth",
        "paddle_path": "retfound_mae_paddle.pdparams",
        "model_type": "mae"
    },
    {
        "pytorch_path": "retfound_dinov2.pth", 
        "paddle_path": "retfound_dinov2_paddle.pdparams",
        "model_type": "dinov2"
    }
]

for config in models_config:
    print(f"转换 {config['model_type']} 模型...")
    convert_retfound_weights(config["pytorch_path"], config["paddle_path"])
```

### 2. 推理优化

```python
# 优化推理性能
@paddle.no_grad()
def optimized_inference(model, images):
    model.eval()
    
    # 批量处理
    batch_size = 32
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        output = model(batch)
        results.append(output)
    
    return paddle.concat(results, axis=0)
```

## 验证清单

使用转换后的权重前，请确认：

- [ ] PyTorch权重文件存在且格式正确
- [ ] 模型配置参数与原始PyTorch模型一致
- [ ] 权重转换过程无错误信息
- [ ] 模型能够正常进行前向传播
- [ ] 输出维度符合预期
- [ ] 推理测试通过（如果启用）

## 环境要求

```bash
# 必需依赖
pip install paddlepaddle-gpu  # 或 paddlepaddle
pip install numpy

# HuggingFace功能（可选）
pip install huggingface_hub
pip install torch  # 仅用于加载PyTorch权重

# 其他工具（可选）
pip install tqdm  # 进度条
```

## 技术支持

如果遇到问题，请按以下步骤检查：

1. **环境依赖** - 确保安装所需包
2. **文件路径** - 检查权重文件路径
3. **模型配置** - 验证模型参数一致性
4. **网络连接** - 检查HuggingFace访问
5. **日志信息** - 查看详细错误日志

相关文件：
- `convert_ckpt.py` - 主要转换脚本
- `util/weight_converter.py` - 权重转换工具
- `models_vit.py` - 模型架构实现 