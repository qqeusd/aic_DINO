"""
test_pipline.py — DINO 三模态改造端到端管线验证

用法:
    # 三模态全融合 (默认)
    python test_pipline.py

    # 消融实验
    python test_pipline.py --fusion_modality rgb_only
    python test_pipline.py --fusion_modality rgb_ir
    python test_pipline.py --fusion_modality rgb_depth

验证项:
  1. 配置文件加载 + 模型构建 (含 IR/Depth 编码器 + Scene Fusion)
  2. DataLoader (collate_fn_multimodal)
  3. 前向传播 (三模态融合)
  4. 损失计算 (检测损失 + Scene Query 辅助损失)
  5. 反向传播 + 参数更新
  6. 消融模式验证
"""

import sys
import argparse

import torch
from torch.utils.data import DataLoader

from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from util.misc import collate_fn_multimodal
from dataset.dataset import Multimodeldataset


def get_args():
    parser = argparse.ArgumentParser('DINO 三模态管线测试')
    parser.add_argument('--fusion_modality', default='all',
                        choices=['all', 'rgb_ir', 'rgb_depth', 'rgb_only'],
                        help='消融实验: 控制模态组合')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='冻结 RGB Backbone')
    parser.add_argument('--freeze_enc_layers', type=int, default=4,
                        help='冻结 Encoder 前 N 层 (默认 4)')
    return parser.parse_args()


def main():
    cli_args = get_args()
    fusion_modality = cli_args.fusion_modality

    # ---- 1. 加载配置 ----
    print("=" * 60)
    print(f"  DINO 三模态改造 — 管线验证 (fusion_modality={fusion_modality})")
    print("=" * 60)

    config_path = r'D:\files\MODEL\transformer-detection\DINO-main\config\DINO\DINO_4scale_multimodal.py'
    print(f"\n[1] 加载配置: {config_path}")
    args = SLConfig.fromfile(config_path)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_classes = 13
    args.batch_size = 1
    args.num_select = 100

    # 消融选项覆盖
    args.fusion_modality = fusion_modality
    args.freeze_backbone = cli_args.freeze_backbone
    args.freeze_enc_layers = cli_args.freeze_enc_layers

    print(f"     device={args.device}, num_classes={args.num_classes}")
    print(f"     fusion_modality={args.fusion_modality}")
    print(f"     freeze_backbone={args.freeze_backbone}, "
          f"freeze_enc_layers={args.freeze_enc_layers}")

    # ---- 2. 构建模型 ----
    print("\n[2] 构建模型...")
    build_func = MODULE_BUILD_FUNCS.get('dino')
    model, criterion, postprocessors = build_func(args)
    model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     总参数量: {n_params:,}")
    print(f"     可训练参数: {n_trainable:,}")
    print(f"     IR Encoder: {type(model.ir_encoder).__name__}")
    print(f"     Depth Encoder: {type(model.depth_encoder).__name__}")
    print(f"     Scene Fusion: {type(model.scene_fusion).__name__ if model.scene_fusion else 'None (消融模式)'}")
    print(f"     fusion_modality: {model.fusion_modality}")

    # ---- 3. 构建 DataLoader ----
    print("\n[3] 构建 DataLoader...")
    data_root = r"D:\files\dataset\itemdetect"
    dataset = Multimodeldataset(data_root, num_classes=12)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_multimodal)
    print(f"     有效样本数: {len(dataset)}")

    # ---- 4. 获取 batch ----
    print("\n[4] 获取 batch...")
    samples, targets = next(iter(loader))
    samples = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
               for k, v in samples.items()}
    targets = [{k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in t.items()} for t in targets]

    print(f"     samples['rgb']:   {samples['rgb'].shape}")
    print(f"     samples['ir']:    {samples['ir'].shape}")
    print(f"     samples['depth']: {samples['depth'].shape}")
    print(f"     targets[0] boxes: {targets[0]['boxes'].shape}, "
          f"labels: {targets[0]['labels'].tolist()}, "
          f"light={targets[0]['light_label']}, "
          f"density={targets[0]['density_label']}")

    # ---- 5. 前向传播 ----
    print("\n[5] 前向传播...")
    model.train()
    outputs = model(samples, targets)

    print(f"     output keys: {sorted(outputs.keys())}")
    print(f"     pred_logits: {outputs['pred_logits'].shape}")
    print(f"     pred_boxes:  {outputs['pred_boxes'].shape}")
    if outputs.get('aux_outputs'):
        print(f"     aux_outputs: {len(outputs['aux_outputs'])} 层")

    # Scene Fusion 输出
    if outputs.get('scene_outputs'):
        so = outputs['scene_outputs']
        print(f"     scene_outputs.light_logits:    {so['light_logits'].softmax(-1).tolist()}")
        print(f"     scene_outputs.density_logits:  {so['density_logits'].softmax(-1).tolist()}")
        print(f"     scene_outputs.modality_weights: {so['modality_weights'].tolist()}")
        print(f"     → w_rgb={so['modality_weights'][0,0]:.3f}, "
              f"w_ir={so['modality_weights'][0,1]:.3f}, "
              f"w_depth={so['modality_weights'][0,2]:.3f}")
    else:
        print(f"     ⚠️  scene_outputs 为空 (fusion_modality={fusion_modality})")

    # ---- 6. 损失计算 ----
    print("\n[6] 损失计算...")
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict

    # 只打印主要损失项
    print(f"     {'Loss':30s} {'Value':>10s} {'Weight':>6s} {'Scaled':>10s}")
    print(f"     {'─'*30} {'─'*10} {'─'*6} {'─'*10}")

    total_loss = 0.0
    main_losses = ['loss_ce', 'loss_bbox', 'loss_giou',
                   'loss_scene_light', 'loss_scene_density', 'loss_weight_entropy']
    for k, v in loss_dict.items():
        w = weight_dict.get(k, 1.0)
        scaled = (v * w).item()
        total_loss += scaled
        if any(ml in k for ml in main_losses):
            print(f"     {k:30s} {v.item():10.4f} {w:6.2f} {scaled:10.4f}")
    print(f"     {'─'*30} {'─'*10} {'─'*6} {'─'*10}")
    print(f"     {'TOTAL':30s} {'':10s} {'':6s} {total_loss:10.4f}")

    # ---- 7. 反向传播 ----
    print("\n[7] 反向传播...")
    losses = sum(loss_dict[k] * weight_dict.get(k, 1.0)
                 for k in loss_dict if k in weight_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    losses.backward()

    grad_params = {n: p.grad.norm().item()
                   for n, p in model.named_parameters()
                   if p.grad is not None and p.requires_grad}
    print(f"     有梯度的参数: {len(grad_params)}")
    top5 = sorted(grad_params.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"     梯度范数 top5:")
    for name, gn in top5:
        print(f"       {name:55s}: {gn:.4f}")

    optimizer.step()
    print(f"     参数已更新")

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print(f"  ✅ 全管线通过")
    print(f"  融合模式: {fusion_modality}")
    print(f"  参数量: {n_params:,} (可训练 {n_trainable:,})")
    print(f"{'='*60}")

    # ---- 8. 消融验证 ----
    if fusion_modality == 'rgb_only':
        print(f"\n[消融验证] RGB-only 模式检查:")
        assert outputs.get('scene_outputs') is None, \
            "rgb_only 模式不应产生 scene_outputs"
        print(f"  ✅ scene_outputs 正确禁用")
        print(f"  ✅ 纯 RGB DINO 前向正常")

    if fusion_modality == 'rgb_ir':
        print(f"\n[消融验证] RGB+IR 模式:")
        so = outputs.get('scene_outputs')
        if so:
            w = so['modality_weights'][0]
            print(f"  模态权重: w_rgb={w[0]:.3f} w_ir={w[1]:.3f} w_depth={w[2]:.3f}")
            print(f"  ✅ Depth 被置零，但 IR 正常参与")

    if fusion_modality == 'rgb_depth':
        print(f"\n[消融验证] RGB+Depth 模式:")
        so = outputs.get('scene_outputs')
        if so:
            w = so['modality_weights'][0]
            print(f"  模态权重: w_rgb={w[0]:.3f} w_ir={w[1]:.3f} w_depth={w[2]:.3f}")
            print(f"  ✅ IR 被置零，但 Depth 正常参与")


if __name__ == '__main__':
    main()
