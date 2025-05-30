#!/usr/bin/env python3
"""
检查PyTorch权重和PaddlePaddle模型参数对齐情况
"""

import os
import paddle
import numpy as np
from models_vit import RETFound_mae
from collections import OrderedDict


def load_pytorch_weights_info(pytorch_model_path):
    """加载并分析PyTorch权重文件信息"""
    try:
        import torch
        print(f"📥 加载PyTorch权重文件: {pytorch_model_path}")
        
        # 加载权重文件
        checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        
        # 处理不同的权重格式
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("权重格式: checkpoint['model']")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("权重格式: checkpoint['state_dict']")
            else:
                state_dict = checkpoint
                print("权重格式: 直接state_dict")
        else:
            state_dict = checkpoint
            print("权重格式: 权重张量")
        
        # 清理权重键名
        cleaned_weights = OrderedDict()
        for key, value in state_dict.items():
            # 移除可能的前缀
            clean_key = key.replace('module.', '').replace('backbone.', '')
            cleaned_weights[clean_key] = value
        
        print(f"🔍 PyTorch权重文件包含 {len(cleaned_weights)} 个参数")
        print(f"📊 总参数量: {sum(v.numel() for v in cleaned_weights.values()):,}")
        
        return cleaned_weights
        
    except Exception as e:
        print(f"❌ 加载PyTorch权重失败: {e}")
        return None


def analyze_paddle_model_structure(model):
    """分析PaddlePaddle模型结构"""
    print(f"🔍 PaddlePaddle模型结构分析:")
    
    model_state_dict = model.state_dict()
    print(f"📊 模型包含 {len(model_state_dict)} 个参数")
    
    # 修复格式化错误
    total_params = sum(p.numel().item() for p in model_state_dict.values())
    print(f"📊 总参数量: {total_params:,}")
    
    return model_state_dict


def detailed_parameter_comparison(pytorch_weights, paddle_state_dict):
    """详细的参数对比分析"""
    print("\n" + "="*60)
    print("📋 详细参数对比分析")
    print("="*60)
    
    # 分类整理参数
    pytorch_keys = set(pytorch_weights.keys())
    paddle_keys = set(paddle_state_dict.keys())
    
    # 找出匹配的参数
    matched_keys = pytorch_keys & paddle_keys
    missing_in_paddle = pytorch_keys - paddle_keys
    missing_in_pytorch = paddle_keys - pytorch_keys
    
    print(f"\n✅ 匹配的参数: {len(matched_keys)}")
    for key in sorted(matched_keys):
        pytorch_shape = pytorch_weights[key].shape
        paddle_shape = paddle_state_dict[key].shape
        
        if pytorch_shape == paddle_shape:
            status = "✅ 形状匹配"
        else:
            status = f"⚠️ 形状不匹配 PyTorch:{pytorch_shape} vs Paddle:{paddle_shape}"
        
        print(f"  {key}: {status}")
    
    print(f"\n❌ PaddlePaddle中缺失的参数: {len(missing_in_paddle)}")
    for key in sorted(missing_in_paddle):
        pytorch_shape = pytorch_weights[key].shape
        print(f"  {key}: {pytorch_shape}")
    
    print(f"\n❌ PyTorch中缺失的参数: {len(missing_in_pytorch)}")
    for key in sorted(missing_in_pytorch):
        paddle_shape = paddle_state_dict[key].shape
        print(f"  {key}: {paddle_shape}")
    
    # 分析缺失参数的类型
    print(f"\n🔍 缺失参数类型分析:")
    
    # 分析decoder相关参数
    decoder_keys = [k for k in missing_in_paddle if 'decoder' in k or 'mask_token' in k]
    if decoder_keys:
        print(f"  🔧 Decoder相关参数 ({len(decoder_keys)}个):")
        for key in decoder_keys[:10]:  # 只显示前10个
            print(f"    {key}")
        if len(decoder_keys) > 10:
            print(f"    ... 还有 {len(decoder_keys)-10} 个decoder参数")
    
    # 分析分类头参数
    head_keys = [k for k in missing_in_paddle if 'head' in k]
    if head_keys:
        print(f"  🎯 分类头相关参数 ({len(head_keys)}个):")
        for key in head_keys:
            print(f"    {key}")
    
    # 分析其他参数
    other_keys = [k for k in missing_in_paddle if 'decoder' not in k and 'mask_token' not in k and 'head' not in k]
    if other_keys:
        print(f"  📦 其他参数 ({len(other_keys)}个):")
        for key in other_keys:
            print(f"    {key}")
    
    return {
        'matched_keys': matched_keys,
        'missing_in_paddle': missing_in_paddle,
        'missing_in_pytorch': missing_in_pytorch,
        'decoder_keys': decoder_keys if 'decoder_keys' in locals() else [],
        'head_keys': head_keys if 'head_keys' in locals() else []
    }


def analyze_mae_vs_classification_model(pytorch_weights):
    """分析MAE模型与分类模型的差异"""
    print(f"\n🔍 MAE vs 分类模型架构分析:")
    
    # 分析encoder参数
    encoder_keys = []
    decoder_keys = []
    head_keys = []
    other_keys = []
    
    for key in pytorch_weights.keys():
        if any(pattern in key for pattern in ['patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm']):
            # 检查是否是decoder块
            if 'decoder' not in key:
                encoder_keys.append(key)
            else:
                decoder_keys.append(key)
        elif 'decoder' in key or 'mask_token' in key:
            decoder_keys.append(key)
        elif 'head' in key:
            head_keys.append(key)
        else:
            other_keys.append(key)
    
    print(f"  🏗️  Encoder参数: {len(encoder_keys)}个")
    print(f"     - patch_embed: {len([k for k in encoder_keys if 'patch_embed' in k])}个")
    print(f"     - cls_token/pos_embed: {len([k for k in encoder_keys if 'cls_token' in k or 'pos_embed' in k])}个")
    print(f"     - transformer blocks: {len([k for k in encoder_keys if 'blocks' in k])}个")
    print(f"     - norm layers: {len([k for k in encoder_keys if 'norm' in k and 'blocks' not in k])}个")
    
    print(f"  🔧 Decoder参数: {len(decoder_keys)}个")
    if decoder_keys:
        print(f"     - decoder_blocks: {len([k for k in decoder_keys if 'decoder_blocks' in k])}个")
        print(f"     - decoder_embed: {len([k for k in decoder_keys if 'decoder_embed' in k])}个")
        print(f"     - decoder_norm: {len([k for k in decoder_keys if 'decoder_norm' in k])}个")
        print(f"     - decoder_pred: {len([k for k in decoder_keys if 'decoder_pred' in k])}个")
        print(f"     - mask_token: {len([k for k in decoder_keys if 'mask_token' in k])}个")
        print(f"     - decoder_pos_embed: {len([k for k in decoder_keys if 'decoder_pos_embed' in k])}个")
    
    print(f"  🎯 分类头参数: {len(head_keys)}个")
    for key in head_keys:
        print(f"     - {key}: {pytorch_weights[key].shape}")
    
    print(f"  📦 其他参数: {len(other_keys)}个")
    for key in other_keys:
        print(f"     - {key}: {pytorch_weights[key].shape}")
    
    return {
        'encoder_keys': encoder_keys,
        'decoder_keys': decoder_keys,
        'head_keys': head_keys,
        'other_keys': other_keys
    }


def suggest_alignment_fixes(comparison_result, mae_analysis):
    """建议参数对齐修复方案"""
    print(f"\n🔧 参数对齐修复建议:")
    
    missing_in_paddle = comparison_result['missing_in_paddle']
    head_keys = mae_analysis['head_keys']
    decoder_keys = mae_analysis['decoder_keys']
    
    # 1. Decoder参数处理
    if decoder_keys:
        print(f"  1️⃣ Decoder参数处理:")
        print(f"     - PyTorch权重包含MAE decoder部分 ({len(decoder_keys)}个参数)")
        print(f"     - PaddlePaddle模型只需要encoder部分")
        print(f"     ✅ 建议: 在权重转换时过滤掉decoder相关参数")
        print(f"     📝 实现: 在weight_converter.py中添加decoder参数过滤")
    
    # 2. 分类头参数处理
    if head_keys:
        print(f"  2️⃣ 分类头参数处理:")
        print(f"     - PyTorch权重包含分类头: {head_keys}")
        for key in head_keys:
            print(f"       {key}: 需要添加到PaddlePaddle模型")
        print(f"     ✅ 建议: 确保PaddlePaddle模型的分类头与PyTorch一致")
    else:
        print(f"  2️⃣ 分类头参数缺失:")
        print(f"     - PyTorch权重缺少分类头参数")
        print(f"     ✅ 建议: 这是正常的，分类头通常需要重新训练")
    
    # 3. 参数键名对齐
    print(f"  3️⃣ 参数键名对齐:")
    print(f"     - 匹配参数: {len(comparison_result['matched_keys'])}个")
    print(f"     - 需要修复的缺失参数: {len(missing_in_paddle - set(decoder_keys))}个")
    
    # 4. 具体修复步骤
    print(f"\n📋 具体修复步骤:")
    print(f"  1. 修改权重转换器以过滤decoder参数")
    print(f"  2. 确保linear层权重正确转置")
    print(f"  3. 处理分类头权重（如果存在）")
    print(f"  4. 验证转换后的权重完整性")


def main():
    """主函数"""
    print("RETFound 参数对齐检查工具")
    print("="*50)
    
    # 1. 加载PyTorch权重信息
    pytorch_model_path = "/home/chenlb24/RETFound_MAE/RETFound_mae_meh.pth"
    
    if not os.path.exists(pytorch_model_path):
        print(f"❌ PyTorch权重文件不存在: {pytorch_model_path}")
        return
    
    pytorch_weights = load_pytorch_weights_info(pytorch_model_path)
    if pytorch_weights is None:
        return
    
    # 2. 创建和分析PaddlePaddle模型
    print(f"\n🏗️  创建PaddlePaddle模型...")
    paddle_model = RETFound_mae(num_classes=1000)
    paddle_state_dict = analyze_paddle_model_structure(paddle_model)
    
    # 3. 详细参数对比
    comparison_result = detailed_parameter_comparison(pytorch_weights, paddle_state_dict)
    
    # 4. MAE vs 分类模型分析
    mae_analysis = analyze_mae_vs_classification_model(pytorch_weights)
    
    # 5. 建议修复方案
    suggest_alignment_fixes(comparison_result, mae_analysis)
    
    print(f"\n✅ 参数对齐检查完成!")


if __name__ == "__main__":
    main() 