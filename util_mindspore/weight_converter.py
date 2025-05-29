"""
PyTorch到MindSpore权重转换工具
"""

import os
import mindspore as ms
from mindspore import Tensor
import numpy as np
from pathlib import Path


def convert_pytorch_to_mindspore_weights(pytorch_checkpoint, model_type='RETFound_mae'):
    """
    将PyTorch权重转换为MindSpore格式
    
    Args:
        pytorch_checkpoint: PyTorch checkpoint字典
        model_type: 模型类型
    
    Returns:
        mindspore_weights: 转换后的MindSpore权重字典
    """
    print("开始转换PyTorch权重到MindSpore格式...")
    
    # 获取模型权重
    if model_type != 'RETFound_mae':
        checkpoint_model = pytorch_checkpoint.get('teacher', pytorch_checkpoint)
    else:
        checkpoint_model = pytorch_checkpoint.get('model', pytorch_checkpoint)
    
    # 清理权重键名
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
    
    # 转换权重格式
    mindspore_weights = {}
    
    for key, value in checkpoint_model.items():
        # 跳过不需要的键
        if key in ['head.weight', 'head.bias']:
            continue
            
        # 转换numpy数组到MindSpore Tensor
        if hasattr(value, 'numpy'):
            # PyTorch tensor
            numpy_value = value.detach().cpu().numpy()
        else:
            # 已经是numpy数组
            numpy_value = value
        
        # 转换为MindSpore参数格式
        mindspore_weights[key] = numpy_value
    
    print(f"成功转换 {len(mindspore_weights)} 个权重参数")
    return mindspore_weights


def save_mindspore_weights(weights_dict, save_path):
    """
    保存MindSpore权重到本地
    
    Args:
        weights_dict: MindSpore权重字典
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为MindSpore参数列表格式
    param_list = []
    for name, data in weights_dict.items():
        param_dict = {
            'name': name,
            'data': Tensor(data, ms.float32)
        }
        param_list.append(param_dict)
    
    # 保存权重
    ms.save_checkpoint(param_list, save_path)
    print(f"MindSpore权重已保存到: {save_path}")


def load_mindspore_weights(checkpoint_path):
    """
    加载MindSpore权重
    
    Args:
        checkpoint_path: 权重文件路径
    
    Returns:
        weights_dict: 权重字典
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        weights_dict = ms.load_checkpoint(checkpoint_path)
        print(f"成功加载MindSpore权重: {checkpoint_path}")
        return weights_dict
    except Exception as e:
        print(f"加载MindSpore权重失败: {e}")
        return None


def download_and_convert_weights(finetune_name, cache_dir="./weights_cache"):
    """
    下载PyTorch权重并转换为MindSpore格式
    
    Args:
        finetune_name: 预训练模型名称
        cache_dir: 缓存目录
    
    Returns:
        mindspore_weights_path: MindSpore权重文件路径
    """
    # 设置缓存路径
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    mindspore_weights_path = cache_dir / f"{finetune_name}_mindspore.ckpt"
    
    # 检查是否已有MindSpore权重
    if mindspore_weights_path.exists():
        print(f"发现已缓存的MindSpore权重: {mindspore_weights_path}")
        return str(mindspore_weights_path)
    
    print(f"未找到MindSpore权重缓存，开始下载并转换PyTorch权重...")
    
    try:
        # 动态导入PyTorch相关模块
        import torch
        from huggingface_hub import hf_hub_download
        
        print(f"从HuggingFace下载预训练权重: {finetune_name}")
        
        # 下载PyTorch权重
        pytorch_checkpoint_path = hf_hub_download(
            repo_id=f'YukunZhou/{finetune_name}',
            filename=f'{finetune_name}.pth',
        )
        
        # 加载PyTorch权重
        pytorch_checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')
        print(f"成功加载PyTorch权重: {pytorch_checkpoint_path}")
        
        # 转换权重格式
        model_type = 'RETFound_mae' if 'mae' in finetune_name else 'RETFound_dinov2'
        mindspore_weights = convert_pytorch_to_mindspore_weights(
            pytorch_checkpoint, model_type
        )
        
        # 保存MindSpore权重
        save_mindspore_weights(mindspore_weights, str(mindspore_weights_path))
        
        return str(mindspore_weights_path)
        
    except ImportError as e:
        print(f"导入PyTorch失败: {e}")
        print("请安装PyTorch和huggingface_hub: pip install torch huggingface_hub")
        raise
    except Exception as e:
        print(f"下载或转换权重失败: {e}")
        raise


def load_pretrained_weights(model, finetune_name, cache_dir="./weights_cache"):
    """
    加载预训练权重到MindSpore模型
    
    Args:
        model: MindSpore模型
        finetune_name: 预训练模型名称
        cache_dir: 缓存目录
    
    Returns:
        model: 加载权重后的模型
    """
    if not finetune_name:
        print("未指定预训练模型，跳过权重加载")
        return model
    
    try:
        # 尝试加载或下载转换权重
        mindspore_weights_path = download_and_convert_weights(finetune_name, cache_dir)
        
        # 加载MindSpore权重
        weights_dict = load_mindspore_weights(mindspore_weights_path)
        
        if weights_dict is None:
            print("加载权重失败")
            return model
        
        # 处理位置编码插值
        from .pos_embed import interpolate_pos_embed
        interpolate_pos_embed(model, weights_dict)
        
        # 加载权重到模型
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, weights_dict, strict_load=False)
        
        if param_not_load:
            print(f"模型中未加载的参数: {param_not_load}")
        if ckpt_not_load:
            print(f"权重文件中未使用的参数: {ckpt_not_load}")
        
        print("预训练权重加载完成")
        
        # 初始化分类头
        if hasattr(model, 'head') and hasattr(model.head, 'weight'):
            # 使用截断正态分布初始化分类头
            from mindspore.common.initializer import TruncatedNormal, initializer
            model.head.weight.set_data(
                initializer(TruncatedNormal(sigma=2e-5), model.head.weight.shape, ms.float32)
            )
            print("分类头权重已重新初始化")
        
        return model
        
    except Exception as e:
        print(f"加载预训练权重失败: {e}")
        print("将使用随机初始化的权重继续训练")
        return model


def check_weight_compatibility(pytorch_state_dict, mindspore_model):
    """
    检查PyTorch权重与MindSpore模型的兼容性
    
    Args:
        pytorch_state_dict: PyTorch状态字典
        mindspore_model: MindSpore模型
    
    Returns:
        compatible_keys: 兼容的键列表
        incompatible_keys: 不兼容的键列表
    """
    model_params = {name: param for name, param in mindspore_model.parameters_and_names()}
    
    compatible_keys = []
    incompatible_keys = []
    
    for key, value in pytorch_state_dict.items():
        if key in model_params:
            model_shape = model_params[key].shape
            pytorch_shape = value.shape if hasattr(value, 'shape') else value.size()
            
            if model_shape == pytorch_shape:
                compatible_keys.append(key)
            else:
                incompatible_keys.append((key, pytorch_shape, model_shape))
        else:
            incompatible_keys.append((key, "not found", "N/A"))
    
    return compatible_keys, incompatible_keys 