#!/usr/bin/env python3
"""
RETFound PyTorch权重加载到PaddlePaddle示例脚本

此脚本演示了如何将RETFound的PyTorch预训练权重转换并加载到PaddlePaddle模型中。
支持从HuggingFace Hub或本地文件加载权重。
"""

import os
import paddle
import numpy as np
from models_vit import RETFound_mae, RETFound_dinov2
from util.weight_converter import convert_retfound_weights, load_converted_weights_to_model


def download_huggingface_model(model_name, cache_dir="./models"):
    """
    从HuggingFace Hub下载RETFound模型权重
    
    Args:
        model_name: 模型名称，如 "ycxia/RETFound_MAE"
        cache_dir: 模型缓存目录
    """
    try:
        from huggingface_hub import hf_hub_download
        import torch
        
        print(f"Downloading {model_name} from HuggingFace Hub...")
        
        # 下载模型权重文件
        model_file = hf_hub_download(
            repo_id=model_name,
            filename="pytorch_model.bin",  # 或 "model.safetensors"
            cache_dir=cache_dir
        )
        
        print(f"Model downloaded to: {model_file}")
        return model_file
        
    except ImportError:
        print("请安装huggingface_hub: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载模型失败: {e}")
        return None


def load_retfound_mae_example():
    """RETFound MAE模型加载示例"""
    print("=== RETFound MAE 模型加载示例 ===")
    
    # 1. 创建PaddlePaddle模型
    model = RETFound_mae(num_classes=1000)
    print(f"Created RETFound MAE model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. 选择权重来源
    pytorch_model_path = None
    
    # 选项1: 从HuggingFace下载
    use_huggingface = True  # 设置为True使用HuggingFace，False使用本地文件
    
    if use_huggingface:
        pytorch_model_path = download_huggingface_model("ycxia/RETFound_MAE")
    else:
        # 选项2: 使用本地PyTorch权重文件
        pytorch_model_path = "path/to/your/pytorch_retfound_mae.pth"
        
        if not os.path.exists(pytorch_model_path):
            print(f"本地文件不存在: {pytorch_model_path}")
            print("请下载RETFound MAE权重或设置正确的文件路径")
            return None
    
    if pytorch_model_path is None:
        print("无法获取PyTorch模型权重")
        return None
    
    # 3. 转换并加载权重
    try:
        success = model.load_pytorch_weights(pytorch_model_path)
        
        if success:
            print("✅ 权重加载成功！")
            
            # 4. 测试模型推理
            test_model_inference(model)
            
            return model
        else:
            print("❌ 权重加载失败")
            return None
            
    except Exception as e:
        print(f"加载权重时出错: {e}")
        return None


def test_model_inference(model):
    """测试模型推理"""
    print("\n=== 测试模型推理 ===")
    
    # 创建随机输入
    batch_size = 2
    input_tensor = paddle.randn([batch_size, 3, 224, 224])
    
    # 设置评估模式
    model.eval()
    
    with paddle.no_grad():
        # 前向传播
        output = model(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 获取特征
        features = model.forward_features(input_tensor)
        print(f"特征形状: {features.shape}")
        
    print("✅ 模型推理测试通过！")


def manual_weight_conversion_example():
    """手动权重转换示例"""
    print("\n=== 手动权重转换示例 ===")
    
    # PyTorch权重文件路径
    pytorch_model_path = "path/to/pytorch_model.pth"
    paddle_model_path = "converted_retfound_paddle.pdparams"
    
    if not os.path.exists(pytorch_model_path):
        print(f"PyTorch权重文件不存在: {pytorch_model_path}")
        return False
    
    # 转换权重
    print("开始转换PyTorch权重...")
    success = convert_retfound_weights(pytorch_model_path, paddle_model_path)
    
    if success:
        print(f"✅ 权重转换成功，保存至: {paddle_model_path}")
        
        # 加载到模型
        model = RETFound_mae(num_classes=1000)
        load_success = load_converted_weights_to_model(model, paddle_model_path, strict=False)
        
        if load_success:
            print("✅ 权重加载到模型成功！")
            return True
        else:
            print("❌ 权重加载到模型失败")
            return False
    else:
        print("❌ 权重转换失败")
        return False


def compare_models_output():
    """比较PyTorch和PaddlePaddle模型输出（如果两个框架都可用）"""
    print("\n=== 模型输出对比 ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # 这里需要有PyTorch版本的RETFound模型
        # 由于我们主要在PaddlePaddle环境中，这个功能作为示例
        print("如果有PyTorch版本，可以在这里进行输出对比")
        print("确保两个模型使用相同的输入和权重")
        
    except ImportError:
        print("PyTorch未安装，跳过模型对比")


def main():
    """主函数"""
    print("RETFound PyTorch权重转换和加载示例")
    print("=" * 50)
    
    # 示例1: 加载RETFound MAE模型
    model = load_retfound_mae_example()
    
    if model is not None:
        print("\n🎉 RETFound模型已成功加载到PaddlePaddle!")
        print("\n你现在可以使用这个模型进行:")
        print("- 微调训练")
        print("- 特征提取") 
        print("- 眼底图像分析")
        print("- 迁移学习")
        
        # 保存PaddlePaddle格式的模型
        paddle_model_save_path = "retfound_mae_paddle.pdparams"
        paddle.save(model.state_dict(), paddle_model_save_path)
        print(f"\n💾 PaddlePaddle模型已保存至: {paddle_model_save_path}")
        
    else:
        print("\n❌ 模型加载失败，请检查权重文件和转换过程")


if __name__ == "__main__":
    # 设置PaddlePaddle设备
    paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    print(f"使用设备: {paddle.get_device()}")
    
    main() 