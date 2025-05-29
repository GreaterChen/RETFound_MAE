#!/usr/bin/env python3
"""
权重转换示例脚本
演示如何将PyTorch权重转换为MindSpore格式
"""

import argparse
from util_mindspore.weight_converter import download_and_convert_weights, load_pretrained_weights
import models_vit_mindspore as models


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to MindSpore format')
    parser.add_argument('--finetune', type=str, required=True,
                        help='Pre-trained model name (e.g., RETFound_mae_natureCFP)')
    parser.add_argument('--cache_dir', type=str, default='./weights_cache',
                        help='Directory to cache converted weights')
    parser.add_argument('--model', type=str, default='RETFound_mae',
                        help='Model type')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyTorch到MindSpore权重转换示例")
    print("=" * 60)
    
    # 1. 下载并转换权重
    print(f"\n1. 下载并转换权重: {args.finetune}")
    try:
        mindspore_weights_path = download_and_convert_weights(
            args.finetune, 
            cache_dir=args.cache_dir
        )
        print(f"✓ 权重转换成功，保存到: {mindspore_weights_path}")
    except Exception as e:
        print(f"✗ 权重转换失败: {e}")
        return
    
    # 2. 创建模型并加载权重
    print(f"\n2. 创建{args.model}模型并加载权重")
    try:
        model = models.__dict__[args.model](
            img_size=224,
            num_classes=args.num_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )
        
        model = load_pretrained_weights(
            model, 
            args.finetune, 
            cache_dir=args.cache_dir
        )
        print("✓ 模型创建和权重加载成功")
        
        # 计算参数数量
        total_params = sum(p.size for p in model.get_parameters())
        print(f"  - 总参数数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
    except Exception as e:
        print(f"✗ 模型创建或权重加载失败: {e}")
        return
    
    # 3. 测试前向传播
    print(f"\n3. 测试前向传播")
    try:
        import mindspore as ms
        from mindspore import Tensor
        import numpy as np
        
        # 创建随机输入
        input_tensor = Tensor(np.random.randn(2, 3, 224, 224), ms.float32)
        
        # 前向传播
        model.set_train(False)
        output = model(input_tensor)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {input_tensor.shape}")
        print(f"  - 输出形状: {output.shape}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ 权重转换和测试完成！")
    print("=" * 60)
    print(f"\n转换后的MindSpore权重文件位置:")
    print(f"  {mindspore_weights_path}")
    print(f"\n使用方法:")
    print(f"  python main_finetune_mindspore.py --finetune {args.finetune} --weights_cache_dir {args.cache_dir} [其他参数...]")


if __name__ == "__main__":
    main() 