# RETFound PyTorch权重加载到PaddlePaddle指南

## 概述

本指南详细说明了如何将RETFound项目的PyTorch预训练权重转换并加载到PaddlePaddle模型中。我们的实现确保了与原始PyTorch模型的完全兼容性。

## 关键特性

✅ **完全兼容** - 模型架构与PyTorch版本完全一致  
✅ **自动转换** - 内置权重转换工具  
✅ **HuggingFace支持** - 直接从HuggingFace Hub加载  
✅ **验证工具** - 确保转换准确性的测试工具  

## 快速开始

### 1. 基本用法

```python
import paddle
from models_vit import RETFound_mae

# 创建模型
model = RETFound_mae(num_classes=1000)

# 从PyTorch权重文件加载
model.load_pytorch_weights("path/to/pytorch_retfound_mae.pth")

# 或从HuggingFace Hub加载
model.load_pytorch_weights("ycxia/RETFound_MAE")  # 自动下载
```

### 2. 完整示例

```python
# 运行完整的示例脚本
python load_pytorch_weights_example.py
```

## 支持的模型

| 模型名称 | HuggingFace Hub | 架构参数 |
|---------|----------------|----------|
| RETFound_MAE | `ycxia/RETFound_MAE` | ViT-Large, 1024d, 24层 |
| RETFound_DINOv2 | `ycxia/RETFound_cf_dinov2` | ViT-Large, 1024d, 24层, patch=14 |

## 模型架构对比

### 与PyTorch实现的一致性

我们的PaddlePaddle实现确保了以下方面与PyTorch完全一致：

#### 1. 层级结构
```
VisionTransformer
├── patch_embed (PatchEmbed)
│   └── proj (Conv2D)
├── cls_token (Parameter)
├── pos_embed (Parameter) 
├── blocks (LayerList)
│   ├── norm1 (LayerNorm)
│   ├── attn (Attention)
│   │   ├── qkv (Linear)
│   │   └── proj (Linear)
│   ├── norm2 (LayerNorm)
│   └── mlp (Mlp)
│       ├── fc1 (Linear)
│       └── fc2 (Linear)
├── norm (LayerNorm)
└── head (Linear)
```

#### 2. 参数命名映射

| PyTorch | PaddlePaddle | 转换说明 |
|---------|-------------|----------|
| `patch_embed.proj.weight` | `patch_embed.proj.weight` | 自动转置维度 |
| `blocks.0.attn.qkv.weight` | `blocks.0.attn.qkv.weight` | 线性层权重转置 |
| `cls_token` | `cls_token` | 直接复制 |
| `pos_embed` | `pos_embed` | 直接复制 |
| `norm.weight/bias` | `norm.weight/bias` | 直接复制 |

#### 3. 关键差异处理

**线性层权重**
- PyTorch: `[out_features, in_features]`
- PaddlePaddle: `[in_features, out_features]`
- 解决方案: 自动转置权重矩阵

**卷积层权重**
- PyTorch: `[out_channels, in_channels, H, W]`  
- PaddlePaddle: `[out_channels, in_channels, H, W]`
- 解决方案: 维度顺序一致，直接复制

## 权重转换工具

### 核心转换函数

```python
from util.weight_converter import convert_retfound_weights

# 转换PyTorch权重到PaddlePaddle格式
success = convert_retfound_weights(
    pytorch_model_path="retfound_mae.pth",
    paddle_model_path="retfound_mae_paddle.pdparams"
)
```

### 自动处理的转换

1. **权重矩阵转置** - 所有线性层权重
2. **参数名称映射** - 自动匹配对应层
3. **数据类型转换** - PyTorch tensor → PaddlePaddle tensor
4. **设备处理** - 自动移动到正确设备

## 验证和测试

### 1. 权重加载验证

```python
# 检查权重是否正确加载
def verify_weights_loading(model, pytorch_weights_path):
    original_state = model.state_dict()
    
    # 加载PyTorch权重
    model.load_pytorch_weights(pytorch_weights_path)
    new_state = model.state_dict()
    
    # 检查权重变化
    for name, param in new_state.items():
        if name in original_state:
            if not paddle.equal(param, original_state[name]):
                print(f"✅ {name}: 权重已更新")
            else:
                print(f"⚠️  {name}: 权重未变化")
```

### 2. 模型输出测试

```python
# 测试模型推理
def test_model_inference(model):
    model.eval()
    
    # 创建测试输入
    x = paddle.randn([1, 3, 224, 224])
    
    with paddle.no_grad():
        # 测试分类输出
        output = model(x)
        print(f"分类输出形状: {output.shape}")  # [1, num_classes]
        
        # 测试特征提取
        features = model.forward_features(x)
        print(f"特征输出形状: {features.shape}")  # [1, embed_dim]
```

### 3. 数值精度验证

如果同时有PyTorch和PaddlePaddle环境，可以进行数值对比：

```python
def compare_outputs(pytorch_model, paddle_model, input_data):
    """比较两个模型的输出"""
    import torch
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_output = pytorch_model(torch_input)
    
    # PaddlePaddle推理  
    paddle_model.eval()
    with paddle.no_grad():
        paddle_input = paddle.to_tensor(input_data)
        paddle_output = paddle_model(paddle_input)
    
    # 计算差异
    diff = np.abs(torch_output.numpy() - paddle_output.numpy())
    print(f"最大差异: {diff.max()}")
    print(f"平均差异: {diff.mean()}")
```

## 常见问题与解决方案

### Q1: 权重加载失败

**可能原因:**
- PyTorch权重文件格式不支持
- 模型架构不匹配
- 权重文件损坏

**解决方案:**
```python
# 检查PyTorch权重文件
import torch
checkpoint = torch.load("model.pth", map_location='cpu')
print(f"权重文件包含的键: {checkpoint.keys()}")

# 如果是完整的checkpoint
if 'model' in checkpoint:
    weights = checkpoint['model']
elif 'state_dict' in checkpoint:
    weights = checkpoint['state_dict']
else:
    weights = checkpoint
```

### Q2: 维度不匹配

**可能原因:**
- 模型配置参数不一致
- 分类头大小不匹配

**解决方案:**
```python
# 创建与PyTorch权重匹配的模型
model = RETFound_mae(
    num_classes=1000,  # 确保与PyTorch模型一致
    global_pool=False  # 检查全局池化设置
)
```

### Q3: HuggingFace下载失败

**解决方案:**
```python
# 设置镜像源
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或手动下载
from huggingface_hub import hf_hub_download
model_file = hf_hub_download(
    repo_id="ycxia/RETFound_MAE",
    filename="pytorch_model.bin",
    cache_dir="./models"
)
```

## 性能优化建议

### 1. 内存优化

```python
# 在转换过程中及时释放内存
import gc
import paddle

def optimized_weight_loading(model, pytorch_path):
    # 加载权重
    model.load_pytorch_weights(pytorch_path)
    
    # 清理内存
    gc.collect()
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
```

### 2. 批量处理

```python
# 对于多个模型的批量转换
models_to_convert = [
    ("retfound_mae.pth", "retfound_mae_paddle.pdparams"),
    ("retfound_dinov2.pth", "retfound_dinov2_paddle.pdparams")
]

for pytorch_path, paddle_path in models_to_convert:
    convert_retfound_weights(pytorch_path, paddle_path)
```

## 进阶用法

### 1. 自定义权重映射

```python
# 如果需要自定义权重名称映射
from util.weight_converter import PyTorchToPaddleConverter

converter = PyTorchToPaddleConverter()
# 添加自定义映射规则
converter.add_mapping_rule(
    pytorch_pattern=r"custom_layer\.(\d+)\.weight",
    paddle_pattern=r"custom_layer.\1.weight"
)
```

### 2. 部分权重加载

```python
# 只加载部分权重（如backbone）
def load_backbone_only(model, pytorch_path):
    from util.weight_converter import convert_retfound_weights_partial
    
    # 指定要转换的层
    layers_to_convert = [
        "patch_embed",
        "cls_token", 
        "pos_embed",
        "blocks"
    ]
    
    convert_retfound_weights_partial(
        pytorch_path, 
        "backbone_only.pdparams",
        include_layers=layers_to_convert
    )
    
    model.set_state_dict(paddle.load("backbone_only.pdparams"), strict=False)
```

## 验证清单

在使用转换后的权重前，请确保：

- [ ] 模型架构参数与原始PyTorch模型一致
- [ ] 权重文件成功转换且无错误信息
- [ ] 模型能够正常进行前向传播
- [ ] 输出维度符合预期
- [ ] 如有条件，数值精度在可接受范围内

## 技术支持

如果遇到问题，请检查：

1. **环境依赖** - 确保安装了所需的包
2. **文件路径** - 检查权重文件路径是否正确
3. **模型配置** - 验证模型参数与PyTorch版本一致
4. **日志信息** - 查看详细的错误日志

更多技术细节请参考：
- `util/weight_converter.py` - 权重转换实现
- `models_vit.py` - 模型架构实现
- `load_pytorch_weights_example.py` - 使用示例 