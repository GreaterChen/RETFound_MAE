#!/usr/bin/env python3
"""
测试MindSpore版本的RETFound实现
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np

# 设置MindSpore上下文
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    
    try:
        from models_vit_mindspore import RETFound_mae
        
        # 创建模型
        model = RETFound_mae(
            img_size=224,
            num_classes=5,
            drop_path_rate=0.1,
            global_pool=True
        )
        
        print(f"✓ 模型创建成功")
        print(f"  - 模型类型: {type(model)}")
        
        # 计算参数数量
        total_params = sum(p.size for p in model.get_parameters())
        print(f"  - 总参数数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n测试前向传播...")
    
    try:
        # 创建随机输入
        batch_size = 2
        input_tensor = Tensor(np.random.randn(batch_size, 3, 224, 224), ms.float32)
        
        # 前向传播
        model.set_train(False)
        output = model(input_tensor)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {input_tensor.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出数据类型: {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False


def test_components():
    """测试各个组件"""
    print("\n测试各个组件...")
    
    try:
        from models_vit_mindspore import PatchEmbed, Attention, Mlp, Block
        
        # 测试PatchEmbed
        patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        x = Tensor(np.random.randn(2, 3, 224, 224), ms.float32)
        patches = patch_embed(x)
        print(f"✓ PatchEmbed: {x.shape} -> {patches.shape}")
        
        # 测试Attention
        attention = Attention(dim=768, num_heads=12)
        x = Tensor(np.random.randn(2, 197, 768), ms.float32)
        attn_out = attention(x)
        print(f"✓ Attention: {x.shape} -> {attn_out.shape}")
        
        # 测试Mlp
        mlp = Mlp(in_features=768, hidden_features=3072)
        mlp_out = mlp(x)
        print(f"✓ Mlp: {x.shape} -> {mlp_out.shape}")
        
        # 测试Block
        block = Block(dim=768, num_heads=12, mlp_ratio=4.0)
        block_out = block(x)
        print(f"✓ Block: {x.shape} -> {block_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 组件测试失败: {e}")
        return False


def test_utilities():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        from util_mindspore.pos_embed import get_2d_sincos_pos_embed
        from util_mindspore.lr_sched import CosineAnnealingLR
        
        # 测试位置编码
        pos_embed = get_2d_sincos_pos_embed(embed_dim=768, grid_size=14)
        print(f"✓ 位置编码: shape={pos_embed.shape}")
        
        # 测试学习率调度
        lr_scheduler = CosineAnnealingLR(base_lr=1e-3, min_lr=1e-6, total_steps=1000, warmup_steps=100)
        lr = lr_scheduler(Tensor(500, ms.int32))
        print(f"✓ 学习率调度: lr={lr}")
        
        return True
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        from engine_finetune_mindspore import LabelSmoothingCrossEntropy
        
        # 测试标签平滑交叉熵
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # 创建测试数据
        logits = Tensor(np.random.randn(4, 5), ms.float32)
        targets = Tensor([0, 1, 2, 3], ms.int32)
        
        loss = criterion(logits, targets)
        print(f"✓ 标签平滑交叉熵: loss={loss}")
        
        return True
        
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("RETFound MindSpore版本转换测试")
    print("=" * 50)
    
    # 测试模型创建
    model = test_model_creation()
    if model is None:
        return
    
    # 测试前向传播
    if not test_forward_pass(model):
        return
    
    # 测试各个组件
    if not test_components():
        return
    
    # 测试工具函数
    if not test_utilities():
        return
    
    # 测试损失函数
    if not test_loss_functions():
        return
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过！MindSpore转换成功！")
    print("=" * 50)


if __name__ == "__main__":
    main() 