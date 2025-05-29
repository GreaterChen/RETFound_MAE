# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import paddle
from paddle.vision import transforms, datasets
from paddle.vision.transforms import Compose
import numpy as np


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_transform(is_train, args):
    # ImageNet默认均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # 训练时的数据增强
    if is_train == 'train':
        transform_list = [
            transforms.Resize(size=(args.input_size + 32, args.input_size + 32)),
            transforms.RandomCrop(size=(args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(prob=0.5),
        ]
        
        # 添加颜色抖动
        if getattr(args, 'color_jitter', None):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=args.color_jitter,
                    contrast=args.color_jitter,
                    saturation=args.color_jitter,
                    hue=args.color_jitter/4
                )
            )
        
        # 添加随机擦除
        if getattr(args, 'reprob', 0) > 0:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                RandomErasing(
                    prob=args.reprob,
                    mode=getattr(args, 'remode', 'pixel'),
                    count=getattr(args, 'recount', 1)
                )
            ])
        else:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        return Compose(transform_list)

    # 验证/测试时的变换
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    
    transform_list = [
        transforms.Resize(size=size),
        transforms.CenterCrop(size=args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    return Compose(transform_list)


class RandomErasing(object):
    """随机擦除数据增强"""
    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, mode='pixel', count=1):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.mode = mode
        self.count = count

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.prob:
            return img

        for _ in range(self.count):
            if self.mode == 'pixel':
                area = img.shape[1] * img.shape[2]
                target_area = np.random.uniform(*self.scale) * area
                aspect_ratio = np.random.uniform(*self.ratio)

                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))

                if w < img.shape[2] and h < img.shape[1]:
                    x1 = np.random.randint(0, img.shape[1] - h)
                    y1 = np.random.randint(0, img.shape[2] - w)
                    if self.value == 'random':
                        img[0, x1:x1+h, y1:y1+w] = np.random.normal(size=(h, w))
                        img[1, x1:x1+h, y1:y1+w] = np.random.normal(size=(h, w))
                        img[2, x1:x1+h, y1:y1+w] = np.random.normal(size=(h, w))
                    else:
                        img[0, x1:x1+h, y1:y1+w] = self.value
                        img[1, x1:x1+h, y1:y1+w] = self.value
                        img[2, x1:x1+h, y1:y1+w] = self.value

        return img
