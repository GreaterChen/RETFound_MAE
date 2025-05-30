# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
import numpy as np


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 确保卷积层权重与PyTorch兼容
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 与PyTorch一致：[B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 确保QKV线性层权重与PyTorch兼容
        # PyTorch: Linear(in_features, out_features) 权重形状为 [out_features, in_features]
        # PaddlePaddle: Linear(in_features, out_features) 权重形状为 [in_features, out_features]
        # 转换工具会处理这个差异
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # QKV投影并重塑为多头格式
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # 计算注意力
        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 应用注意力并重塑
        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)  # binarize
        
        # 确保keep_prob是tensor
        keep_prob_tensor = paddle.to_tensor(keep_prob, dtype=x.dtype)
        output = x.divide(keep_prob_tensor) * random_tensor
        return output


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # 正确的DropPath实现
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Pre-norm结构，与PyTorch ViT一致
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for global average pooling
    与PyTorch实现保持一致的ViT模型
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, epsilon=1e-6), global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token 和 positional embedding
        # 确保初始化方式与PyTorch一致
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], 
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        # 分类头的norm层
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
            
        # 最终分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化模型权重，与PyTorch ViT保持一致"""
        # 初始化patch embedding
        w = self.patch_embed.proj.weight
        paddle.nn.initializer.XavierUniform()(w.reshape([w.shape[0], -1]))
        
        # 初始化cls_token和pos_embed
        paddle.nn.initializer.TruncatedNormal(std=0.02)(self.cls_token)
        paddle.nn.initializer.TruncatedNormal(std=0.02)(self.pos_embed)
        
        # 初始化分类头
        if isinstance(self.head, nn.Linear):
            paddle.nn.initializer.TruncatedNormal(std=0.02)(self.head.weight)
            if self.head.bias is not None:
                paddle.nn.initializer.Constant(0)(self.head.bias)

    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)

        # 添加CLS token
        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat((cls_tokens, x), axis=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 全局平均池化或使用CLS token
        if self.global_pool:
            x = x[:, 1:, :].mean(axis=1)  # 全局平均池化，排除CLS token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # 使用CLS token

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pytorch_weights(self, pytorch_model_path):
        """直接加载PyTorch权重的便捷方法"""
        from util.weight_converter import convert_retfound_weights, load_converted_weights_to_model
        import os
        
        # 转换权重文件名
        base_name = os.path.splitext(os.path.basename(pytorch_model_path))[0]
        paddle_model_path = f"{base_name}_paddle.pdparams"
        
        # 转换权重
        print(f"Converting PyTorch weights from {pytorch_model_path}")
        success = convert_retfound_weights(pytorch_model_path, paddle_model_path)
        
        if success:
            # 加载转换后的权重
            print(f"Loading converted weights into PaddlePaddle model")
            return load_converted_weights_to_model(self, paddle_model_path, strict=False)
        else:
            print("Failed to convert PyTorch weights")
            return False


def RETFound_mae(num_classes=1000, **kwargs):
    """RETFound MAE模型（与HuggingFace模型兼容）"""
    # 设置默认的img_size，但如果kwargs中有，则使用kwargs中的值
    img_size = kwargs.pop('img_size', 224)
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=num_classes,
        **kwargs)
    return model


def RETFound_dinov2(num_classes=1000, **kwargs):
    """RETFound DINOv2模型（与HuggingFace模型兼容）"""
    model = VisionTransformer(
        img_size=224,
        patch_size=14,  # DINOv2使用14x14的patch
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=num_classes,
        **kwargs)
    return model


def vit_base_patch16(num_classes=1000, **kwargs):
    """标准ViT-Base/16模型"""
    model = VisionTransformer(
        img_size=224,
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=num_classes,
        **kwargs)
    return model


def vit_large_patch16(num_classes=1000, **kwargs):
    """标准ViT-Large/16模型"""
    model = VisionTransformer(
        img_size=224,
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=num_classes,
        **kwargs)
    return model



