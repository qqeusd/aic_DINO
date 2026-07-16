"""
三模态目标检测数据集模块

支持 RGB(可见光) + Infrared(热红外) + Depth(深度) 三模态数据加载。
数据目录结构应如下:
    root/
    ├── visible/    # RGB 可见光图像 (.jpg/.png/.jpeg)
    ├── infrared/    # 红外图像 (.jpg/.png/.jpeg)
    ├── depth/       # 深度图 (16位单通道, .png)
    └── labels/      # YOLO 格式标签 (.txt)
"""

import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F_t


class Multimodeldataset:
    """
    三模态目标检测数据集。

    加载 RGB / Infrared / Depth 三模态图像及对应的 YOLO 格式标签，
    返回 resize 到固定尺寸的目标尺寸后的 tensor 数据。

    Attributes:
        root: 数据集根目录路径
        num_classes: 目标类别数（不含背景）
        vis_dir: 可见光图像目录 (root/visible)
        inf_dir: 红外图像目录 (root/infrared)
        depth_dir: 深度图目录 (root/depth)
        label_dir: 标签目录 (root/labels)
        samples: 有效样本的文件名(stem)列表

    Example:
        >>> ds = Multimodeldataset("/path/to/dataset")
        >>> sample = ds[0]
        >>> print(sample['rgb'].shape)      # torch.Size([3, 800, 800])
        >>> print(sample['ir'].shape)       # torch.Size([3, 800, 800])
        >>> print(sample['depth'].shape)    # torch.Size([1, 800, 800])
        >>> print(sample['target'].keys())  # dict_keys(['boxes', 'labels', ...])
    """

    # [新增] 光照阈值常量（用于自动生成 light_label）
    LIGHT_THRESH_BRIGHT = 0.5    # RGB 均值 > 0.5 → bright
    LIGHT_THRESH_DIM = 0.2       # 0.2 < RGB 均值 ≤ 0.5 → dim,  ≤ 0.2 → dark

    # [新增] 密度阈值常量（用于自动生成 density_label）
    DENSITY_THRESH_SPARSE = 3     # GT 框数 ≤ 3 → sparse
    DENSITY_THRESH_DENSE = 8      # 3 < GT 框数 ≤ 8 → medium,  > 8 → dense

    def __init__(self, root, num_classes=12):
        """
        初始化数据集，扫描目录并收集有效样本。

        Args:
            root (str): 数据集根目录，应包含 visible/infrared/depth/labels 四个子目录。
                        只保留四个子目录中均存在同名文件的样本。
            num_classes (int): [新增] 目标类别数（不含背景，默认 12）。
                              用于标签验证和类别相关配置。

        Raises:
            FileNotFoundError: 如果 root 或其子目录不存在。
        """
        self.root = root
        self.num_classes = num_classes  # [新增]
        self.vis_dir = os.path.join(root, 'visible')
        self.inf_dir = os.path.join(root, 'infrared')
        self.depth_dir = os.path.join(root, 'depth')
        self.label_dir = os.path.join(root, 'labels')
        self.samples = self.collect_samples()

    # ------------------------------------------------------------------
    # 样本收集与文件查找
    # ------------------------------------------------------------------

    def collect_samples(self):
        """
        收集所有有效样本的文件名（不含扩展名）。

        遍历 visible 目录，找出在 infrared、depth、labels 目录中
        均存在同名文件的样本 stem。

        Returns:
            list[str]: 有效样本的 stem 列表（按文件名排序）。

        Note:
            只保留在四个目录中同时存在文件的样本，缺失任一模态或标签的
            样本将被静默跳过。
        """
        samples = []
        for filename in sorted(os.listdir(self.vis_dir)):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            stem = os.path.splitext(filename)[0]

            if (self.has_stem(self.inf_dir, stem) and
                self.has_stem(self.depth_dir, stem) and
                self.has_stem(self.label_dir, stem)):
                samples.append(stem)
        return samples

    def has_stem(self, dir, stem):
        """
        检查指定目录中是否存在以 stem 为文件名的文件（任意扩展名）。

        Args:
            dir (str): 要搜索的目录路径。
            stem (str): 文件名（不含扩展名，如 "00000008"）。

        Returns:
            bool: 如果目录中存在该 stem 的文件则返回 True，否则返回 False。
        """
        for filename in os.listdir(dir):
            name = os.path.splitext(filename)[0]
            if name == stem:
                return True
        return False

    def _find_path(self, directory, stem):
        """
        根据 stem 在指定目录中查找文件完整路径。

        Args:
            directory (str): 要搜索的目录路径。
            stem (str): 文件名（不含扩展名）。

        Returns:
            str: 匹配文件的完整路径。

        Raises:
            FileNotFoundError: 如果目录中不存在该 stem 的文件。
        """
        for fname in os.listdir(directory):
            if os.path.splitext(fname)[0] == stem:
                return os.path.join(directory, fname)
        raise FileNotFoundError(f"{stem} not found in {directory}")

    # ------------------------------------------------------------------
    # 单模态加载器
    # ------------------------------------------------------------------

    def load_rgb(self, path):
        """
        加载 RGB 图像并归一化到 [0, 1]。

        Args:
            path (str): RGB 图像文件路径。

        Returns:
            np.ndarray: 形状为 [H, W, 3] 的 float32 数组，数值范围 [0, 1]。
        """
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def load_depth(self, path):
        """
        加载 16 位深度图，处理无效值并归一化。

        处理流程:
            1. 读取图像（保持原始位深，不 convert）
            2. 若为 3 维数组（多通道），只取第 0 通道
            3. 将 depth=0 或 depth>19999 的像素视为无效，置零
            4. 归一化: arr / 20000 → [0, 1]
            5. 扩展最后一维: [H, W] → [H, W, 1]

        Args:
            path (str): 深度图文件路径。

        Returns:
            tuple:
                - arr (np.ndarray): 形状为 [H, W, 1] 的 float32 数组，数值范围 [0, 1]。
                - depth_mask (np.ndarray): 形状为 [H, W] 的 bool 数组，True 表示有效深度值。

        Raises:
            ValueError: 如果深度图维度大于 3。
        """
        img = Image.open(path)

        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            # 防御：多通道深度图只取第 0 通道
            arr = arr[:, :, 0]
        elif arr.ndim > 3:
            raise ValueError(f"深度图维度异常: {arr.shape}")

        # 无效值检测：0 和 ≥19999 视为无效
        depth_mask = (arr > 0) & (arr < 19999)

        # 无效区域置零，有效区域归一化
        arr = np.where(depth_mask, arr, 0)
        arr = arr / 20000.0
        arr = np.expand_dims(arr, -1)  # [H, W] → [H, W, 1]

        return arr, depth_mask

    def load_label(self, path):
        """
        加载 YOLO 格式标签文件。

        标签文件格式（每行一个目标）:
            class_id center_x center_y width height
            所有坐标已归一化到 [0, 1]（相对于原图宽高）。

        Args:
            path (str): 标签文件路径 (.txt)。

        Returns:
            tuple:
                - boxes (torch.Tensor): 形状为 [N, 4] 的 float32 tensor，
                  每行为 (cx, cy, w, h)，归一化坐标。
                - labels (torch.Tensor): 形状为 [N] 的 int64 tensor，
                  每个元素为类别索引。
                - 若无目标，返回 (0,4) 和 (0,) 的空 tensor。
        """
        boxes = []
        labels = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(cls)

        if len(boxes) == 0:
            return (torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.long))

        return (torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long))

    # ------------------------------------------------------------------
    # PyTorch Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self):
        """
        返回数据集样本总数。

        Returns:
            int: 有效样本数量。
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取指定索引的样本。

        加载三模态图像和标签，统一 resize 到 (800, 800)，
        构造 target 字典（兼容 DINO 的 SetCriterion 格式）。

        Args:
            index (int): 样本索引。

        Returns:
            dict:
                - 'rgb' (torch.Tensor): 形状 [3, 800, 800]，float32，范围 [0, 1]
                - 'ir' (torch.Tensor): 形状 [3, 800, 800]，float32，范围 [0, 1]
                - 'depth' (torch.Tensor): 形状 [1, 800, 800]，float32，范围 [0, 1]
                - 'target' (dict):
                    - 'boxes' (torch.Tensor): [N, 4] 归一化 cxcywh 坐标
                    - 'labels' (torch.Tensor): [N] 类别索引 (int64)
                    - 'image_id' (torch.Tensor): [1] 图像索引
                    - 'area' (torch.Tensor): [N] 归一化面积 (w*h)
                    - 'iscrowd' (torch.Tensor): [N] 全零 (int64)
                    - 'orig_size' (torch.Tensor): [2] 原始图像尺寸 [H, W]
                    - 'size' (torch.Tensor): [2] 当前尺寸 [H, W]
                    - 'depth_mask' (torch.Tensor): [800, 800] 有效深度掩码 (float32)
                    - 'stem' (str): 样本文件名（不含扩展名）
        """
        stem = self.samples[index]

        # ---- 1. 查找各模态文件路径 ----
        vis_path = self._find_path(self.vis_dir, stem)
        depth_path = self._find_path(self.depth_dir, stem)
        label_path = self._find_path(self.label_dir, stem)
        inf_path = self._find_path(self.inf_dir, stem)

        # ---- 2. 加载各模态数据 ----
        visible = self.load_rgb(vis_path)               # [H, W, 3], float32, [0,1]
        infrared = self.load_rgb(inf_path)              # [H, W, 3], float32, [0,1]
        depth, depth_mask = self.load_depth(depth_path)  # [H, W, 1], [H, W]
        h, w = visible.shape[:2]                         # 原始图像尺寸
        boxes, labels = self.load_label(label_path)      # [N, 4], [N]

        # ---- 2.5 [新增] 标签验证 ----
        if len(boxes) > 0:
            # 检查 boxes 坐标是否在 [0, 1] 归一化范围内
            if not (0 <= boxes[:, 0]).all() or not (boxes[:, 0] <= 1).all():
                raise ValueError(
                    f"[{stem}] boxes cx 超出 [0,1] 范围: "
                    f"min={boxes[:, 0].min().item():.4f}, max={boxes[:, 0].max().item():.4f}"
                )
            if not (0 <= boxes[:, 1]).all() or not (boxes[:, 1] <= 1).all():
                raise ValueError(
                    f"[{stem}] boxes cy 超出 [0,1] 范围: "
                    f"min={boxes[:, 1].min().item():.4f}, max={boxes[:, 1].max().item():.4f}"
                )
            if not (0 <= boxes[:, 2]).all() or not (boxes[:, 2] <= 1).all():
                raise ValueError(
                    f"[{stem}] boxes w 超出 [0,1] 范围: "
                    f"min={boxes[:, 2].min().item():.4f}, max={boxes[:, 2].max().item():.4f}"
                )
            if not (0 <= boxes[:, 3]).all() or not (boxes[:, 3] <= 1).all():
                raise ValueError(
                    f"[{stem}] boxes h 超出 [0,1] 范围: "
                    f"min={boxes[:, 3].min().item():.4f}, max={boxes[:, 3].max().item():.4f}"
                )
            # 检查类别索引是否在合法范围内
            if not (0 <= labels).all() or not (labels < self.num_classes).all():
                bad_labels = labels[(labels < 0) | (labels >= self.num_classes)]
                raise ValueError(
                    f"[{stem}] 类别标签超出 [0, {self.num_classes - 1}] 范围: {bad_labels.tolist()}"
                )

        # ---- 3. 转为 Tensor 并调整维度顺序 ----
        visible_t = torch.from_numpy(visible).permute(2, 0, 1).float()    # [3, H, W]
        infrared_t = torch.from_numpy(infrared).permute(2, 0, 1).float()  # [3, H, W]
        depth_t = torch.from_numpy(depth).permute(2, 0, 1).float()        # [1, H, W]
        depth_mask_t = torch.from_numpy(depth_mask.astype(np.float32))     # [H, W]

        # ---- 4. 统一 Resize ----
        target_size = (800, 800)

        visible_t = F_t.resize(visible_t, target_size)                     # [3, 800, 800]
        infrared_t = F_t.resize(infrared_t, target_size)                   # [3, 800, 800]
        depth_t = F_t.resize(depth_t, target_size)                         # [1, 800, 800]
        depth_mask_t = F_t.resize(
            depth_mask_t.unsqueeze(0),
            target_size,
            interpolation=F_t.InterpolationMode.NEAREST                     # 最近邻保持 0/1
        ).squeeze(0)                                                        # [800, 800]

        # ---- 5. 构造 target 字典（兼容 DINO SetCriterion） ----
        target = {
            'boxes': boxes,                                    # [N, 4] 归一化 cxcywh
            'labels': labels,                                  # [N]    类别 id (int64)
            'image_id': torch.tensor([index]),                 # [1]    图像索引
            'area': boxes[:, 2] * boxes[:, 3],                 # [N]    归一化面积 (w*h)
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64),  # [N]
            'orig_size': torch.tensor([h, w]),                 # [2]    原始尺寸 [H, W]
            'size': torch.tensor(target_size),                 # [2]    当前尺寸
            'depth_mask': depth_mask_t,                        # [800, 800] 有效深度掩码
            'stem': stem,                                      # str    文件名
        }

        # ---- 5.5 [新增] 自动生成辅助标签（Scene Query 监督信号） ----
        # 光照标签：基于 RGB 均值
        rgb_mean = visible_t.mean().item()
        if rgb_mean > self.LIGHT_THRESH_BRIGHT:
            light_label = 0  # bright
        elif rgb_mean > self.LIGHT_THRESH_DIM:
            light_label = 1  # dim
        else:
            light_label = 2  # dark

        # 密度标签：基于 GT 框数量
        num_boxes = len(labels)
        if num_boxes <= self.DENSITY_THRESH_SPARSE:
            density_label = 0  # sparse
        elif num_boxes <= self.DENSITY_THRESH_DENSE:
            density_label = 1  # medium
        else:
            density_label = 2  # dense

        target['light_label'] = light_label      # int, {0: bright, 1: dim, 2: dark}
        target['density_label'] = density_label  # int, {0: sparse, 1: medium, 2: dense}

        # ---- 6. 组装样本 ----
        sample = {
            'rgb': visible_t,       # [3, 800, 800]
            'ir': infrared_t,       # [3, 800, 800]
            'depth': depth_t,       # [1, 800, 800]
            'target': target,
        }
        return sample
