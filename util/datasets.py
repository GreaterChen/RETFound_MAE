# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import paddle
from paddle.vision import transforms
from paddle.vision.transforms import Compose
from paddle.io import Dataset
import numpy as np
from PIL import Image


class ImageFolderDataset(Dataset):
    """自定义的图像文件夹数据集，确保与ImageFolder格式兼容"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}
        
        # 扫描目录结构
        self._make_dataset()
        
    def _make_dataset(self):
        """扫描目录并创建样本列表"""
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"目录不存在: {self.root_dir}")
            
        # 获取所有类别文件夹
        class_names = [d for d in os.listdir(self.root_dir) 
                      if os.path.isdir(os.path.join(self.root_dir, d))]
        class_names.sort()  # 确保顺序一致
        
        self.class_names = class_names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # 扫描每个类别文件夹中的图像
        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if self._is_image_file(filename):
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, class_idx))
        
        print(f"数据集加载完成: {len(self.samples)} 个样本, {len(self.class_names)} 个类别")
        print(f"类别映射: {self.class_to_idx}")
        
    def _is_image_file(self, filename):
        """判断是否为图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # 加载图像
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回一个空白图像作为fallback
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_idx


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    
    # 使用自定义的ImageFolderDataset
    dataset = ImageFolderDataset(root, transform=transform)
    
    print(f"构建 {is_train} 数据集: {len(dataset)} 个样本")
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
