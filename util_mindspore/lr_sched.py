# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Converted to MindSpore framework
# --------------------------------------------------------

import math
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    # Update learning rate for MindSpore optimizer
    if hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate.set_data(ms.Tensor(lr, ms.float32))
    else:
        # For optimizers that don't support dynamic learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lr


class CosineAnnealingLR(LearningRateSchedule):
    """Cosine annealing learning rate schedule for MindSpore"""
    
    def __init__(self, base_lr, min_lr, total_steps, warmup_steps=0):
        super(CosineAnnealingLR, self).__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def construct(self, global_step):
        if global_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * global_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + ms.ops.cos(math.pi * progress))
        
        return lr


class WarmupCosineAnnealingLR(LearningRateSchedule):
    """Warmup + Cosine annealing learning rate schedule"""
    
    def __init__(self, base_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super(WarmupCosineAnnealingLR, self).__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
    
    def construct(self, global_step):
        if global_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * global_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + ms.ops.cos(math.pi * progress))
        
        return lr


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.parameters_and_names():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % '\n'.join(['%s: %d' % (k, len(v["params"])) for k, v in param_group_names.items()]))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1


def create_optimizer_with_layer_decay(model, lr, weight_decay, layer_decay, filter_bias_and_bn=True):
    """Create optimizer with layer-wise learning rate decay"""
    
    # Get parameter groups with layer decay
    param_groups = param_groups_lrd(model, weight_decay=weight_decay, layer_decay=layer_decay)
    
    # Apply learning rate scaling to each group
    for group in param_groups:
        group['lr'] = lr * group['lr_scale']
    
    # Create AdamW optimizer
    optimizer = nn.AdamWeightDecay(
        params=param_groups,
        learning_rate=lr,
        weight_decay=weight_decay
    )
    
    return optimizer 