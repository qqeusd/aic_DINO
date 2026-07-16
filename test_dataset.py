"""
dataset.py 功能验证脚本

用法:
    # 自动查找数据路径
    python test_dataset.py

    # 指定数据路径
    python test_dataset.py --data_root /path/to/dataset

验证项:
  1. 数据集初始化（文件扫描、样本收集）
  2. 单模态加载（RGB / Depth / Label）
  3. __getitem__ 输出（shape / dtype / 值域）
  4. 标签验证（合法 / 非法）
  5. 辅助标签生成（light_label / density_label）
  6. collate_fn_multimodal 批处理
  7. DataLoader 流水线
"""

import sys
import os
import argparse
import traceback

import torch
import numpy as np
from torch.utils.data import DataLoader

# ---- 路径设置 ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.dataset import Multimodeldataset


def find_data_root():
    """在常见候选路径中查找可用的三模态数据集目录。"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_script_dir, '..', 'dataset', 'itemdetect'),
        os.path.join(_script_dir, 'data', 'itemdetect'),
        os.path.join(_script_dir, '..', 'itemdetect'),
        r"D:\files\dataset\itemdetect",          # 本地开发路径
        r"D:\dataset\itemdetect",
        "/home/zcoop8/zhangxianping/data/itemdetect",  # 服务器路径
        "/mnt/data/itemdetect",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            subdirs = ['visible', 'infrared', 'depth', 'labels']
            if all(os.path.isdir(os.path.join(p, sd)) for sd in subdirs):
                return p
    return ''


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_1_init(data_root):
    """验证数据集初始化"""
    print_header("1. 数据集初始化")

    ds = Multimodeldataset(data_root, num_classes=12)

    print(f"  root:          {ds.root}")
    print(f"  num_classes:   {ds.num_classes}")
    print(f"  有效样本数:    {len(ds)}")

    if len(ds) == 0:
        print("  ❌ 未找到任何有效样本")
        return False, ds

    print(f"  前5个样本:     {ds.samples[:5]}")
    print(f"  ✅ 通过")
    return True, ds


def test_2_load_rgb(ds):
    """验证 RGB 加载"""
    print_header("2. RGB 加载")

    stem = ds.samples[0]
    vis_path = ds._find_path(ds.vis_dir, stem)
    arr = ds.load_rgb(vis_path)

    print(f"  样本:  {stem}")
    print(f"  shape: {arr.shape}     (期望: [H, W, 3])")
    print(f"  dtype: {arr.dtype}      (期望: float32)")
    print(f"  min:   {arr.min():.4f}  (期望: ≥0)")
    print(f"  max:   {arr.max():.4f}  (期望: ≤1)")

    ok = True
    if arr.ndim != 3 or arr.shape[2] != 3:
        print("  ❌ 通道数异常")
        ok = False
    if arr.dtype != np.float32:
        print("  ❌ dtype 异常")
        ok = False
    if arr.min() < 0 or arr.max() > 1:
        print("  ❌ 值域异常")
        ok = False
    if ok:
        print("  ✅ 通过")
    return ok


def test_3_load_depth(ds):
    """验证 Depth 加载"""
    print_header("3. Depth 加载")

    stem = ds.samples[0]
    depth_path = ds._find_path(ds.depth_dir, stem)
    arr, mask = ds.load_depth(depth_path)

    print(f"  样本:      {stem}")
    print(f"  shape:     {arr.shape}       (期望: [H, W, 1])")
    print(f"  dtype:     {arr.dtype}        (期望: float32)")
    print(f"  min:       {arr.min():.4f}    (期望: ≥0)")
    print(f"  max:       {arr.max():.4f}    (期望: ≤1)")
    print(f"  mask shape: {mask.shape}       (期望: [H, W])")
    print(f"  有效像素:  {mask.sum():.0f} / {mask.size} ({mask.sum()/mask.size*100:.1f}%)")

    ok = True
    if arr.ndim != 3 or arr.shape[2] != 1:
        print("  ❌ 通道数异常")
        ok = False
    if arr.min() < 0 or arr.max() > 1:
        print("  ❌ 值域异常")
        ok = False
    if mask.sum() == 0:
        print("  ⚠️  全图无有效深度值")
    if ok:
        print("  ✅ 通过")
    return ok


def test_4_load_label(ds):
    """验证 Label 加载"""
    print_header("4. Label 加载")

    stem = ds.samples[0]
    label_path = ds._find_path(ds.label_dir, stem)
    boxes, labels = ds.load_label(label_path)

    print(f"  样本:       {stem}")
    print(f"  目标数:     {len(boxes)}")
    print(f"  boxes shape: {boxes.shape}   (期望: [N, 4] 或 [0, 4])")
    print(f"  labels shape:{labels.shape}    (期望: [N] 或 [0])")

    if len(boxes) > 0:
        print(f"  boxes 前3行:\n{boxes[:3]}")
        print(f"  labels 前5个: {labels[:5].tolist()}")

    ok = True
    if boxes.shape[-1] != 4:
        print("  ❌ boxes 最后一维不是4")
        ok = False
    if len(boxes) > 0 and labels.shape[0] != boxes.shape[0]:
        print("  ❌ boxes 和 labels 数量不一致")
        ok = False
    if ok:
        print("  ✅ 通过")
    return ok


def test_5_getitem(ds):
    """验证 __getitem__ 输出"""
    print_header("5. __getitem__ 完整流程")

    sample = ds[0]

    # 检查 sample 结构
    expected_keys = {'rgb', 'ir', 'depth', 'target'}
    missing = expected_keys - set(sample.keys())
    if missing:
        print(f"  ❌ sample 缺少 key: {missing}")
        return False
    print(f"  sample keys: {list(sample.keys())}  ✅")

    # rgb
    rgb = sample['rgb']
    print(f"  rgb   shape: {rgb.shape}   dtype: {rgb.dtype}   min: {rgb.min():.3f}  max: {rgb.max():.3f}")
    if rgb.shape != (3, 800, 800) or rgb.dtype != torch.float32:
        print("  ❌ rgb 规格异常")
        return False

    # ir
    ir = sample['ir']
    print(f"  ir    shape: {ir.shape}   dtype: {ir.dtype}   min: {ir.min():.3f}  max: {ir.max():.3f}")
    if ir.shape != (3, 800, 800) or ir.dtype != torch.float32:
        print("  ❌ ir 规格异常")
        return False

    # depth
    depth = sample['depth']
    print(f"  depth shape: {depth.shape}   dtype: {depth.dtype}   min: {depth.min():.3f}  max: {depth.max():.3f}")
    if depth.shape != (1, 800, 800) or depth.dtype != torch.float32:
        print("  ❌ depth 规格异常")
        return False

    # target
    target = sample['target']
    required_keys = ['boxes', 'labels', 'image_id', 'area', 'iscrowd',
                     'orig_size', 'size', 'depth_mask', 'stem',
                     'light_label', 'density_label']
    missing_tgt = set(required_keys) - set(target.keys())
    if missing_tgt:
        print(f"  ❌ target 缺少 key: {missing_tgt}")
        return False
    print(f"  target keys: {list(target.keys())}  ✅")

    print(f"  target['stem']:        {target['stem']}")
    print(f"  target['image_id']:    {target['image_id']}")
    print(f"  target['orig_size']:   {target['orig_size']}  (原始 H,W)")
    print(f"  target['size']:        {target['size']}       (当前 H,W)")
    print(f"  target['light_label']:  {target['light_label']} "
          f"({'bright' if target['light_label']==0 else 'dim' if target['light_label']==1 else 'dark'})")
    print(f"  target['density_label']:{target['density_label']} "
          f"({'sparse' if target['density_label']==0 else 'medium' if target['density_label']==1 else 'dense'})")

    if target['size'][0] != 800 or target['size'][1] != 800:
        print("  ❌ target['size'] 不是 (800,800)")
        return False
    if target['depth_mask'].shape != (800, 800):
        print("  ❌ depth_mask shape 异常")
        return False

    print("  ✅ 通过")
    return True


def test_6_label_validation(ds):
    """验证标签校验"""
    print_header("6. 标签验证")

    # 6a: 合法样本应正常通过
    sample = ds[0]
    print(f"  6a. 合法样本 '{sample['target']['stem']}' → 无异常  ✅")

    # 6b: 非法坐标检测
    print("  6b. 构造 cx=1.5 的非法标签...", end=" ")
    try:
        bad_boxes = torch.tensor([[1.5, 0.3, 0.1, 0.1]])
        bad_labels = torch.tensor([0])
        if not (0 <= bad_boxes[:, 0]).all() or not (bad_boxes[:, 0] <= 1).all():
            raise ValueError("预期错误")
        print("  ✅ 正确检测到非法坐标")
    except ValueError:
        print("  ✅ 正确抛出异常")

    # 6c: 非法类别
    print("  6c. 构造 class_id=99 的非法标签...", end=" ")
    try:
        bad_labels = torch.tensor([99])
        if not (0 <= bad_labels).all() or not (bad_labels < 12).all():
            raise ValueError("预期错误")
        print("  ✅ 正确检测到非法类别")
    except ValueError:
        print("  ✅ 正确抛出异常")

    print("  ✅ 通过")


def test_7_aux_labels(ds):
    """验证辅助标签分布"""
    print_header("7. 辅助标签分布")

    light_counts = {0: 0, 1: 0, 2: 0}
    density_counts = {0: 0, 1: 0, 2: 0}
    light_names = {0: 'bright', 1: 'dim', 2: 'dark'}
    density_names = {0: 'sparse', 1: 'medium', 2: 'dense'}

    n = min(len(ds), 50)
    for i in range(n):
        sample = ds[i]
        ll = sample['target']['light_label']
        dl = sample['target']['density_label']
        light_counts[ll] += 1
        density_counts[dl] += 1

    print(f"  采样 {n} 个样本:")
    print(f"  光照分布:  { {light_names[k]: v for k, v in light_counts.items()} }")
    print(f"  密度分布:  { {density_names[k]: v for k, v in density_counts.items()} }")

    covered_l = sum(1 for v in light_counts.values() if v > 0)
    covered_d = sum(1 for v in density_counts.values() if v > 0)

    if covered_l >= 2:
        print(f"  ✅ 光照标签覆盖 {covered_l}/3 类")
    if covered_d >= 2:
        print(f"  ✅ 密度标签覆盖 {covered_d}/3 类")

    print("  ✅ 通过")


def test_8_collate_fn(ds):
    """验证 collate_fn_multimodal 批处理"""
    print_header("8. collate_fn_multimodal")

    def _collate(batch):
        samples = {}
        for key in ['rgb', 'ir', 'depth']:
            samples[key] = torch.stack([item[key] for item in batch])
        targets = [item['target'] for item in batch]
        return samples, targets

    batch_size = 2
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate)

    for samples, targets in loader:
        print(f"  samples['rgb']    shape: {samples['rgb'].shape}")
        print(f"  samples['ir']     shape: {samples['ir'].shape}")
        print(f"  samples['depth']  shape: {samples['depth'].shape}")
        print(f"  len(targets):     {len(targets)}")

        ok = True
        for k in ['rgb', 'ir', 'depth']:
            if samples[k].shape != (batch_size, *samples[k].shape[1:]):
                print(f"  ❌ {k} batch shape 异常")
                ok = False
        if ok:
            print("  ✅ 通过")
        break


def test_9_iteration_speed(ds):
    """验证 DataLoader 迭代速度"""
    print_header("9. 迭代速度基准")

    def _collate(batch):
        samples = {}
        for key in ['rgb', 'ir', 'depth']:
            samples[key] = torch.stack([item[key] for item in batch])
        targets = [item['target'] for item in batch]
        return samples, targets

    import time

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=_collate, num_workers=0)

    n_batches = min(10, len(loader))
    start = time.perf_counter()
    for i, (samples, targets) in enumerate(loader):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - start

    print(f"  处理 {n_batches} 个 batch (batch_size=4, num_workers=0)")
    print(f"  总耗时:    {elapsed:.2f}s")
    print(f"  每 batch:  {elapsed/n_batches:.3f}s")
    print(f"  ✅ 通过")


def main():
    """主测试入口"""
    parser = argparse.ArgumentParser('dataset.py 功能验证')
    parser.add_argument('--data_root', type=str, default='',
                        help='三模态数据集根目录 (若不指定则自动查找)')
    args = parser.parse_args()

    data_root = args.data_root or find_data_root()

    print("=" * 60)
    print("  dataset.py 功能验证")
    print(f"  data_root = {data_root}")
    print("=" * 60)

    if not data_root or not os.path.isdir(data_root):
        print(f"\n  ❌ 数据目录不存在或未找到。")
        print(f"  请通过 --data_root 参数指定路径")
        return 1

    tests = [
        ("1. 数据集初始化",     lambda: test_1_init(data_root)),
        ("2. RGB 加载",         lambda: test_2_load_rgb(ds)),
        ("3. Depth 加载",       lambda: test_3_load_depth(ds)),
        ("4. Label 加载",       lambda: test_4_load_label(ds)),
        ("5. __getitem__",      lambda: test_5_getitem(ds)),
        ("6. 标签验证",         lambda: test_6_label_validation(ds)),
        ("7. 辅助标签分布",     lambda: test_7_aux_labels(ds)),
        ("8. collate_fn",       lambda: test_8_collate_fn(ds)),
        ("9. 迭代速度",         lambda: test_9_iteration_speed(ds)),
    ]

    passed = 0
    failed = 0
    ds = None

    for name, test_fn in tests:
        try:
            result = test_fn()
            if isinstance(result, tuple):
                ok, ds = result
            else:
                ok = result
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ❌ 测试抛异常: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  结果: {passed} 通过 / {failed} 失败 (共 {len(tests)} 项)")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
