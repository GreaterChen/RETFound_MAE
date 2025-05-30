#!/usr/bin/env python3
"""
RETFound PyTorch权重加载到PaddlePaddle示例脚本

此脚本演示了如何将RETFound的PyTorch预训练权重转换并加载到PaddlePaddle模型中。
支持从HuggingFace Hub或本地文件加载权重，支持镜像站和token认证。
"""

import os
import time
import argparse
import paddle
import numpy as np
from models_vit import RETFound_mae, RETFound_dinov2
from util.weight_converter import convert_retfound_weights, load_converted_weights_to_model


def setup_huggingface_config(args):
    """
    设置HuggingFace配置，包括镜像站和token登录
    
    Args:
        args: 命令行参数对象
    """
    try:
        from huggingface_hub import login
        
        # 设置HF镜像站
        hf_endpoint = args.hf_endpoint or os.getenv('HF_ENDPOINT')
        if hf_endpoint:
            print(f"使用 HF 镜像站: {hf_endpoint}")
            os.environ['HF_ENDPOINT'] = hf_endpoint
        
        # 登录 Hugging Face
        hf_token = args.hf_token or os.getenv('HF_TOKEN')
        if hf_token:
            print("正在登录 Hugging Face...")
            login(token=hf_token)
            print("✅ Hugging Face 登录成功")
        else:
            print("⚠️  警告: 未提供 Hugging Face token，可能无法访问受限仓库")
            print("你可以使用 --hf_token 参数或设置 HF_TOKEN 环境变量")
        
        return hf_endpoint
        
    except ImportError:
        print("请安装huggingface_hub: pip install huggingface_hub")
        return None


def download_huggingface_model(model_name, cache_dir="./models", hf_endpoint=None, max_retries=3):
    """
    从HuggingFace Hub下载RETFound模型权重，支持镜像站和重试机制
    
    Args:
        model_name: 模型名称，如 "ycxia/RETFound_MAE"
        cache_dir: 模型缓存目录
        hf_endpoint: HuggingFace镜像站地址
        max_retries: 最大重试次数
    """
    try:
        from huggingface_hub import hf_hub_download
        import torch
        
        print(f"正在从 HuggingFace Hub 下载 {model_name}...")
        
        # 下载模型权重文件，支持重试机制
        for attempt in range(max_retries):
            try:
                download_kwargs = {
                    'repo_id': model_name,
                    'filename': "pytorch_model.bin",  # 或尝试其他可能的文件名
                    'cache_dir': cache_dir
                }
                
                # 如果指定了镜像站地址，添加到参数中
                if hf_endpoint:
                    download_kwargs['endpoint'] = hf_endpoint
                
                model_file = hf_hub_download(**download_kwargs)
                print(f"✅ 模型下载成功: {model_file}")
                return model_file
                
            except Exception as e:
                print(f"下载尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                
                # 如果是文件名问题，尝试其他可能的文件名
                if "does not exist" in str(e) and "pytorch_model.bin" in str(e):
                    try:
                        download_kwargs['filename'] = "model.safetensors"
                        model_file = hf_hub_download(**download_kwargs)
                        print(f"✅ 使用 safetensors 格式下载成功: {model_file}")
                        return model_file
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    print("所有下载尝试都失败了，请尝试:")
                    print("1. 使用 --hf_endpoint https://hf-mirror.com")
                    print("2. 设置环境变量: export HF_ENDPOINT=https://hf-mirror.com")
                    print("3. 检查网络连接")
                    print("4. 确认模型名称是否正确")
                    raise e
                else:
                    print(f"5秒后重试...")
                    time.sleep(5)
        
        return None
        
    except ImportError:
        print("请安装huggingface_hub: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载模型失败: {e}")
        return None


def load_retfound_mae_example(args):
    """RETFound MAE模型加载示例"""
    print("=== RETFound MAE 模型加载示例 ===")
    
    # 1. 创建PaddlePaddle模型
    model = RETFound_mae(num_classes=5)
    print(f"创建 RETFound MAE 模型，参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 2. 选择权重来源
    pytorch_model_path = None
    
    # 检查是否使用本地模式
    if not args.use_huggingface:  # use_local模式
        print("📁 使用本地pth文件模式")
        
        if not args.local_model_path:
            print("❌ 错误: 本地模式需要指定 --local_model_path 参数")
            print("使用方式: python script.py --use_local --local_model_path /path/to/model.pth")
            return None
        
        pytorch_model_path = args.local_model_path
        
        # 检查文件是否存在
        if not os.path.exists(pytorch_model_path):
            print(f"❌ 模型文件不存在: {pytorch_model_path}")
            return None
        
        if not os.path.isfile(pytorch_model_path):
            print(f"❌ 指定路径不是文件: {pytorch_model_path}")
            return None
        
        print(f"✅ 找到模型文件: {pytorch_model_path}")
            
    else:  # HuggingFace下载模式
        print("🌐 使用 HuggingFace 下载模式")
        # 设置HuggingFace配置（包括登录等）
        hf_endpoint = setup_huggingface_config(args)
        pytorch_model_path = download_huggingface_model(
            args.model_name, 
            cache_dir=args.cache_dir,
            hf_endpoint=hf_endpoint,
            max_retries=args.max_retries
        )
    
    if pytorch_model_path is None:
        print("无法获取PyTorch模型权重")
        return None
    
    # 3. 转换并加载权重
    try:
        print(f"🔄 开始加载权重: {pytorch_model_path}")
        success = model.load_pytorch_weights(pytorch_model_path)
        
        if success:
            print("✅ 权重加载成功！")
            
            # 4. 测试模型推理
            if args.test_inference:
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RETFound PyTorch权重转换和加载工具')
    
    # HuggingFace 相关参数
    parser.add_argument('--hf_token', default=None, type=str, 
                        help='Hugging Face token for accessing gated repositories')
    parser.add_argument('--hf_endpoint', default="https://hf-mirror.com", type=str,
                        help='Hugging Face endpoint URL (use https://hf-mirror.com for China mirror)')
    
    # 模型相关参数
    parser.add_argument('--model_name', default="YukunZhou/RETFound_mae_meh", type=str,
                        help='HuggingFace model repository name')
    parser.add_argument('--cache_dir', default="./models", type=str,
                        help='Directory to cache downloaded models')
    parser.add_argument('--local_model_path', type=str, default="/home/chenlb24/RETFound_MAE/RETFound_mae_meh.pth",
                        help='Path to local PyTorch model file')
    
    # 下载和处理参数
    parser.add_argument('--max_retries', default=3, type=int,
                        help='Maximum number of download retries')
    parser.add_argument('--use_huggingface', action='store_true', default=False,
                        help='Download from HuggingFace Hub (default: True)')
    parser.add_argument('--use_local', dest='use_huggingface', action='store_false', default=True,
                        help='Use local model file instead of downloading')
    
    # 功能选项
    parser.add_argument('--test_inference', action='store_true', default=True,
                        help='Test model inference after loading')
    parser.add_argument('--validation_samples', default=5, type=int,
                        help='Number of samples for validation testing')
    parser.add_argument('--save_paddle_model', action='store_true', default=False,
                        help='Save converted model in PaddlePaddle format (default: False, since model.load_pytorch_weights already saves converted weights)')
    parser.add_argument('--paddle_model_path', default=None, type=str,
                        help='Path to save PaddlePaddle model (default: auto-generated based on input filename)')
    
    # 设备参数
    parser.add_argument('--device', default='auto', choices=['auto', 'gpu', 'cpu'],
                        help='Device to use for computation')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("RETFound PyTorch权重转换和加载示例")
    print("=" * 50)
    print("📝 注意: 权重转换过程说明")
    print("   1. model.load_pytorch_weights() 会自动转换并保存权重文件")
    print("   2. 如果启用 --save_paddle_model，脚本会检查是否已存在文件，避免重复保存")
    print("   3. 默认情况下只保存一份转换后的权重文件")
    print("=" * 50)
    print()
    
    # 显示当前模式信息
    if not args.use_huggingface:  # 本地模式
        print("🏠 模式: 本地文件模式")
        if args.local_model_path:
            print(f"📁 本地模型路径: {args.local_model_path}")
        else:
            print("⚠️  未指定本地模型路径")
    else:  # HuggingFace模式
        print("🌐 模式: HuggingFace 下载模式")
        print(f"📦 模型名称: {args.model_name}")
        print(f"📂 缓存目录: {args.cache_dir}")
        if args.hf_endpoint:
            print(f"🌍 HF镜像站: {args.hf_endpoint}")
    
    # 设置PaddlePaddle设备
    if args.device == 'auto':
        device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    else:
        device = args.device
    
    paddle.set_device(device)
    print(f"🖥️  使用设备: {paddle.get_device()}")
    print()
    
    # 示例1: 加载RETFound MAE模型
    model = load_retfound_mae_example(args)
    
    if model is not None:
        print("\n🎉 RETFound模型已成功加载到PaddlePaddle!")
        print("\n你现在可以使用这个模型进行:")
        print("- 微调训练")
        print("- 特征提取") 
        print("- 眼底图像分析")
        print("- 迁移学习")
        
        # 保存PaddlePaddle格式的模型
        if args.save_paddle_model:
            # 如果没有指定保存路径，则根据输入文件自动生成
            if args.paddle_model_path is None:
                if not args.use_huggingface and args.local_model_path:
                    base_name = os.path.splitext(os.path.basename(args.local_model_path))[0]
                    args.paddle_model_path = f"{base_name}_paddle.pdparams"
                else:
                    args.paddle_model_path = "retfound_mae_paddle.pdparams"
            
            paddle.save(model.state_dict(), args.paddle_model_path)
            print(f"💾 PaddlePaddle模型已保存至: {args.paddle_model_path}")
        
    else:
        if not args.use_huggingface:
            print("\n❌ 本地模型加载失败，请检查:")
            print("1. 文件路径是否正确")
            print("2. 模型文件是否存在")
            print("3. 文件格式是否正确")
        else:
            print("\n❌ HuggingFace模型下载/加载失败，请检查:")
            print("1. 网络连接")
            print("2. 模型名称是否正确")
            print("3. 权重文件和转换过程")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 