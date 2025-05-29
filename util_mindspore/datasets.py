# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# Converted to MindSpore framework
# --------------------------------------------------------

import os
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
import numpy as np


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    
    # Create MindSpore dataset from ImageFolder
    dataset = ds.ImageFolderDataset(
        dataset_dir=root,
        shuffle=(is_train == 'train'),
        num_parallel_workers=args.num_workers
    )
    
    # Apply transforms
    dataset = dataset.map(
        operations=transform,
        input_columns=["image"],
        num_parallel_workers=args.num_workers
    )
    
    # Batch the dataset
    if is_train == 'train':
        dataset = dataset.batch(
            batch_size=args.batch_size,
            drop_remainder=True
        )
    else:
        dataset = dataset.batch(
            batch_size=args.batch_size,
            drop_remainder=False
        )
    
    return dataset


def build_transform(is_train, args):
    # Default ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if args.norm == 'IMAGENET':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Train transform
    if is_train == 'train':
        transform_list = [
            vision.Decode(),
            vision.Resize(size=(args.input_size, args.input_size), interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(prob=0.5),
        ]
        
        # Add color jitter if specified
        if args.color_jitter is not None:
            transform_list.append(
                vision.RandomColorAdjust(
                    brightness=args.color_jitter,
                    contrast=args.color_jitter,
                    saturation=args.color_jitter,
                    hue=args.color_jitter/2
                )
            )
        
        # Add random erasing if specified
        if args.reprob > 0:
            transform_list.append(
                vision.RandomErasing(
                    prob=args.reprob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False
                )
            )
        
        # Convert to tensor and normalize
        transform_list.extend([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False)
        ])
        
    else:
        # Validation/test transform
        transform_list = [
            vision.Decode(),
            vision.Resize(size=(args.input_size, args.input_size), interpolation=Inter.BICUBIC),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False)
        ]
    
    return transform_list


def create_transform(
    input_size,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment=None,
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    tf_preprocessing=False,
    separate=False,
):
    """
    MindSpore equivalent of timm's create_transform function
    """
    
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = (input_size, input_size)
    
    if is_training and not no_aug:
        # Training transforms
        transform_list = [
            vision.Decode(),
            vision.Resize(size=img_size, interpolation=Inter.BICUBIC),
        ]
        
        if hflip > 0.:
            transform_list.append(vision.RandomHorizontalFlip(prob=hflip))
        
        if vflip > 0.:
            transform_list.append(vision.RandomVerticalFlip(prob=vflip))
        
        if color_jitter is not None and color_jitter > 0:
            transform_list.append(
                vision.RandomColorAdjust(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter/2
                )
            )
        
        # Random erasing
        if re_prob > 0.:
            transform_list.append(
                vision.RandomErasing(
                    prob=re_prob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False
                )
            )
        
    else:
        # Validation transforms
        transform_list = [
            vision.Decode(),
            vision.Resize(size=img_size, interpolation=Inter.BICUBIC),
        ]
    
    # Convert to tensor and normalize
    transform_list.extend([
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std, is_hwc=False)
    ])
    
    return transform_list


class Mixup:
    """MindSpore implementation of Mixup data augmentation"""
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None,
                 prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.mixup_enabled = mixup_alpha > 0. or cutmix_alpha > 0. or cutmix_minmax is not None

    def __call__(self, x, target):
        if not self.mixup_enabled:
            return x, target
        
        # This is a simplified implementation
        # Full implementation would require more complex logic
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0. else 1.
        
        if np.random.rand() < self.mix_prob:
            # Simple mixup
            batch_size = x.shape[0]
            indices = np.random.permutation(batch_size)
            x = lam * x + (1 - lam) * x[indices]
            
            # Mix targets (this would need proper one-hot encoding)
            target_a, target_b = target, target[indices]
            return x, (target_a, target_b, lam)
        
        return x, target 