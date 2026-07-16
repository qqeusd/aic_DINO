"""
三模态数据集构建模块

为 DINO 训练流水线提供三模态数据集的 build 函数，
与 datasets/__init__.py 中的 build_dataset 接口保持一致。

用法:
    args.dataset_file = 'multimodal'
    args.coco_path = '/path/to/multimodal/root'

    然后在 main.py 中调用 build_dataset(image_set, args) 即可。
"""

import os
import sys

import torch
from torch.utils.data import Subset

# dataset/dataset.py 位于项目根目录的 dataset/ 子目录下
# 确保路径可导入
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_cur_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataset.dataset import Multimodeldataset


def build_multimodal(image_set, args):
    """
    构建三模态数据集的 train 或 val 子集。

    从 args.coco_path 下读取数据，默认按 90%/10% 划分 train/val，
    可通过 args.train_ratio 覆盖划分比例。

    Args:
        image_set (str): 'train' 或 'val'。
        args (argparse.Namespace): 全局配置参数，必须包含:
            - coco_path (str): 数据集根目录路径。
            - num_classes (int): 目标类别数（不含背景），默认 12。
            - train_ratio (float, 可选): 训练集比例，默认 0.9。

    Returns:
        torch.utils.data.Dataset:
            - image_set='train': 训练集子集。
            - image_set='val':   验证集子集。
            - image_set='test':  全量数据集（用于最终测试）。

    Raises:
        FileNotFoundError: 如果 coco_path 下不存在 visible/infrared/depth/labels 子目录。
        ValueError: 如果 image_set 不是 'train' / 'val' / 'test'。
    """
    root = args.coco_path
    num_classes = getattr(args, 'num_classes', 12)
    train_ratio = getattr(args, 'train_ratio', 0.9)

    # 创建全量数据集
    full_dataset = Multimodeldataset(root, num_classes=num_classes)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"在 {root} 下未找到任何三模态样本。"
            f"请确认目录结构: {root}/{{visible,infrared,depth,labels}}/"
        )

    if image_set == 'test':
        # 测试集: 返回全量数据
        return full_dataset

    if image_set in ('train', 'val'):
        # 固定种子划分，保证 train/val 可复现
        n_total = len(full_dataset)
        n_train = max(1, int(n_total * train_ratio))
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        if image_set == 'train':
            return Subset(full_dataset, train_indices)
        else:
            return Subset(full_dataset, val_indices)

    raise ValueError(f"image_set 必须为 'train'/'val'/'test'，当前: '{image_set}'")
