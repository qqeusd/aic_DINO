from torch.utils.data import DataLoader
import torch
from dataset import Multimodeldataset


def collate_fn(batch):
    """
    batch: list of dict，每个元素是 Dataset.__getitem__ 的返回值
    返回: (samples_dict, targets_list)
    """
    # 把三模态图像分别 stack（它们的尺寸相同，可以直接 stack）
    rgb = torch.stack([b['rgb'] for b in batch])  # [B, 3, H, W]
    ir = torch.stack([b['ir'] for b in batch])  # [B, 3, H, W]
    depth = torch.stack([b['depth'] for b in batch])  # [B, 1, H, W]

    # targets 是变长的（每张图目标数不同），不能 stack，保留为 list
    targets = [b['target'] for b in batch]

    # 打包成字典，作为 "samples" 传入模型
    samples = {
        'rgb': rgb,
        'ir': ir,
        'depth': depth,
    }
    return samples, targets

# 若需测试，请取消注释并指定数据路径：
# dataset = Multimodeldataset("/path/to/your/dataset")
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
# for samples, targets in dataloader:
#     print("rgb:\t", samples['rgb'].shape)
#     print("depth:\t", samples['depth'].shape)
#     print("ir:\t", samples['ir'].shape)
#     print("targets:\t", targets)
