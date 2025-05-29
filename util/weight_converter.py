import torch
import paddle
import numpy as np
from collections import OrderedDict
import re

class PyTorchToPaddleConverter:
    """PyTorch模型权重转换到PaddlePaddle格式"""
    
    def __init__(self):
        # 定义权重名称映射规则
        self.weight_name_mapping = {
            # LayerNorm权重映射
            'weight': 'weight',
            'bias': 'bias',
            # 线性层权重映射
            'linear.weight': 'weight',
            'linear.bias': 'bias',
            # 卷积层权重映射
            'conv.weight': 'weight',
            'conv.bias': 'bias',
        }
        
        # 需要转置的权重类型
        self.transpose_weights = [
            'qkv.weight',
            'proj.weight', 
            'fc1.weight',
            'fc2.weight',
            'head.weight'
        ]
    
    def convert_weight_name(self, pytorch_name):
        """转换PyTorch权重名称为PaddlePaddle格式"""
        # 基本替换规则
        paddle_name = pytorch_name
        
        # 处理特殊命名规则
        replacements = [
            # Transformer block命名
            ('blocks.', 'blocks.'),
            ('norm1.', 'norm1.'),
            ('norm2.', 'norm2.'),
            ('attn.', 'attn.'),
            ('mlp.', 'mlp.'),
            # 位置编码等
            ('pos_embed', 'pos_embed'),
            ('cls_token', 'cls_token'),
            ('patch_embed.proj.', 'patch_embed.proj.'),
            # 最终分类头
            ('head.', 'head.'),
            ('norm.', 'norm.'),
            ('fc_norm.', 'fc_norm.'),
        ]
        
        for old, new in replacements:
            paddle_name = paddle_name.replace(old, new)
            
        return paddle_name
    
    def convert_weight_value(self, name, weight_tensor):
        """转换权重值格式"""
        # 转换为numpy数组
        if isinstance(weight_tensor, torch.Tensor):
            weight_np = weight_tensor.detach().cpu().numpy()
        else:
            weight_np = weight_tensor
            
        # 检查是否需要转置线性层权重
        # PyTorch线性层权重形状为[out_features, in_features]
        # PaddlePaddle线性层权重形状为[in_features, out_features]
        needs_transpose = any(pattern in name for pattern in self.transpose_weights)
        
        if needs_transpose and len(weight_np.shape) == 2:
            weight_np = weight_np.T
            print(f"Transposed weight {name}: {weight_tensor.shape} -> {weight_np.shape}")
            
        return weight_np
    
    def convert_state_dict(self, pytorch_state_dict):
        """转换完整的state_dict"""
        paddle_state_dict = OrderedDict()
        
        for pytorch_name, pytorch_weight in pytorch_state_dict.items():
            # 转换权重名称
            paddle_name = self.convert_weight_name(pytorch_name)
            
            # 转换权重值
            paddle_weight = self.convert_weight_value(pytorch_name, pytorch_weight)
            
            paddle_state_dict[paddle_name] = paddle_weight
            print(f"Converted: {pytorch_name} -> {paddle_name}, shape: {paddle_weight.shape}")
            
        return paddle_state_dict
    
    def save_paddle_weights(self, paddle_state_dict, save_path):
        """保存PaddlePaddle格式的权重"""
        # 转换为Paddle张量
        paddle_tensors = {}
        for name, weight in paddle_state_dict.items():
            paddle_tensors[name] = paddle.to_tensor(weight, dtype='float32')
            
        paddle.save(paddle_tensors, save_path)
        print(f"Saved converted weights to {save_path}")


def load_pytorch_retfound_weights(pytorch_model_path):
    """加载PyTorch RETFound模型权重"""
    try:
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        
        # 处理不同的权重保存格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 移除可能的前缀
        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # 移除 'module.' 前缀（分布式训练产生的）
            if key.startswith('module.'):
                key = key[7:]
            # 移除其他可能的前缀
            if key.startswith('backbone.'):
                key = key[9:]
            cleaned_state_dict[key] = value
            
        return cleaned_state_dict
        
    except Exception as e:
        print(f"Error loading PyTorch weights: {e}")
        return None


def convert_retfound_weights(pytorch_model_path, paddle_model_path):
    """转换RETFound模型权重从PyTorch到PaddlePaddle"""
    
    print("Loading PyTorch weights...")
    pytorch_state_dict = load_pytorch_retfound_weights(pytorch_model_path)
    
    if pytorch_state_dict is None:
        print("Failed to load PyTorch weights")
        return False
        
    print(f"Loaded {len(pytorch_state_dict)} parameters from PyTorch model")
    
    # 创建转换器
    converter = PyTorchToPaddleConverter()
    
    print("Converting weights...")
    paddle_state_dict = converter.convert_state_dict(pytorch_state_dict)
    
    print("Saving PaddlePaddle weights...")
    converter.save_paddle_weights(paddle_state_dict, paddle_model_path)
    
    print("Weight conversion completed successfully!")
    return True


def load_converted_weights_to_model(model, paddle_model_path, strict=True):
    """将转换后的权重加载到PaddlePaddle模型中"""
    try:
        # 加载权重
        state_dict = paddle.load(paddle_model_path)
        
        # 获取模型的参数名称
        model_state_dict = model.state_dict()
        
        print("Model parameters:")
        for name in model_state_dict.keys():
            print(f"  {name}: {model_state_dict[name].shape}")
            
        print("\nLoaded parameters:")
        for name in state_dict.keys():
            print(f"  {name}: {state_dict[name].shape}")
            
        # 检查参数匹配
        missing_keys = []
        unexpected_keys = []
        
        for name in model_state_dict.keys():
            if name not in state_dict:
                missing_keys.append(name)
                
        for name in state_dict.keys():
            if name not in model_state_dict:
                unexpected_keys.append(name)
                
        if missing_keys:
            print(f"\nMissing keys: {missing_keys}")
            
        if unexpected_keys:
            print(f"\nUnexpected keys: {unexpected_keys}")
            
        # 加载权重
        if strict and (missing_keys or unexpected_keys):
            print("Strict mode enabled but there are missing or unexpected keys")
            return False
        else:
            # 只加载匹配的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_state_dict and 
                                 v.shape == model_state_dict[k].shape}
            
            model.set_state_dict(filtered_state_dict)
            print(f"Successfully loaded {len(filtered_state_dict)} parameters")
            return True
            
    except Exception as e:
        print(f"Error loading weights to model: {e}")
        return False


if __name__ == "__main__":
    # 示例用法
    pytorch_model_path = "path/to/pytorch_retfound_model.pth"
    paddle_model_path = "converted_retfound_paddle.pdparams"
    
    # 转换权重
    success = convert_retfound_weights(pytorch_model_path, paddle_model_path)
    
    if success:
        print("Weight conversion successful!")
        
        # 示例：加载到模型
        from models_vit import RETFound_mae
        model = RETFound_mae(num_classes=1000)
        load_converted_weights_to_model(model, paddle_model_path, strict=False) 