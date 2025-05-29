# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    # PaddlePaddle优化器使用set_lr方法设置学习率
    if hasattr(optimizer, 'set_lr'):
        optimizer.set_lr(lr)
    else:
        # 如果是自定义优化器，直接设置param_groups
        for param_group in optimizer._param_groups:
            if "lr_scale" in param_group:
                param_group["learning_rate"] = lr * param_group["lr_scale"]
            else:
                param_group["learning_rate"] = lr
    return lr
