# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Converted to MindSpore framework
# --------------------------------------------------------

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import math


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).transpose(0, 3, 1, 2)
            
            # Convert to MindSpore tensor for interpolation
            pos_tokens_tensor = Tensor(pos_tokens, ms.float32)
            pos_tokens = ops.interpolate(
                pos_tokens_tensor, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            pos_tokens = pos_tokens.transpose(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
            new_pos_embed = ops.concat((extra_tokens, pos_tokens), axis=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    """
    Rescale the grid of position embeddings when loading from checkpoint.
    Adapted from:
    https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).transpose(0, 3, 1, 2)
    posemb_grid = ops.interpolate(
        Tensor(posemb_grid, ms.float32), 
        size=gs_new, 
        mode='bicubic', 
        align_corners=False
    )
    posemb_grid = posemb_grid.transpose(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = ops.concat([posemb_tok, posemb_grid], axis=1)
    return posemb


class PositionalEncoding(nn.Cell):
    """Positional encoding for Vision Transformer"""
    
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        pe = np.zeros((max_len, embed_dim))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe.reshape(1, max_len, embed_dim)
        
        self.pe = Tensor(pe, ms.float32)
    
    def construct(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


def build_2d_sincos_position_embedding(w, h, embed_dim, temperature=10000.):
    """Build 2D sin-cos position embedding"""
    grid_w = ops.arange(int(w), dtype=ms.float32)
    grid_h = ops.arange(int(h), dtype=ms.float32)
    grid_w, grid_h = ops.meshgrid((grid_w, grid_h), indexing='ij')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    
    pos_dim = embed_dim // 4
    omega = ops.arange(pos_dim, dtype=ms.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    
    out_w = ops.einsum('m,d->md', grid_w.flatten(), omega)
    out_h = ops.einsum('m,d->md', grid_h.flatten(), omega)
    
    pos_emb = ops.concat([
        ops.sin(out_w), ops.cos(out_w),
        ops.sin(out_h), ops.cos(out_h)
    ], axis=1)[None, :, :]
    
    return pos_emb 