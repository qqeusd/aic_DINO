"""
DINO 三模态训练配置

基于 DINO_4scale.py，覆盖以下参数以适应三模态微调:
    - dataset_file = 'multimodal'
    - num_classes = 13 (12类 + ∅)
    - lr_backbone = 0 (冻结 RGB Backbone)
    - 50 epoch 三阶段训练
    - 新模块 (ir_encoder/depth_encoder/scene_fusion) 用高学习率
    - Transformer 用低学习率微调

消融实验:
    --fusion_modality all      三模态融合 (默认)
    --fusion_modality rgb_ir   RGB + IR
    --fusion_modality rgb_depth RGB + Depth
    --fusion_modality rgb_only 纯 RGB (跳过融合模块)
"""

_base_ = ['DINO_4scale.py']

# ============================================================
# 数据集
# ============================================================
dataset_file = 'multimodal'
num_classes = 13
dn_labelbook_size = 13          # 必须 >= num_classes

# ============================================================
# 训练策略 (三阶段)
# ============================================================
epochs = 50
lr_drop = 20                    # StepLR: epoch 20 衰减
multi_step_lr = True            # 改用 MultiStepLR
lr_drop_list = [20, 40]         # epoch 20: 0.1x, epoch 40: 再 0.1x

# ============================================================
# 分层学习率
# ============================================================
param_dict_type = 'multimodal'  # 自定义分层策略
lr = 1e-4                       # 新模块高学习率
lr_backbone = 1e-10             # backbone 会被 freeze_backbone 冻结，此处仅满足安全检查
lr_multimodal_backbone = 0      # backbone LR (0 = 冻结)
lr_multimodal_transformer = 1e-5  # Transformer 低学习率微调

# ============================================================
# 冻结策略
# ============================================================
freeze_backbone = True          # 始终冻结 RGB Backbone
freeze_enc_layers = 4           # Encoder 前 4 层冻结, 后 2 层微调

# ============================================================
# 消融选项: 通过命令行 --fusion_modality 覆盖
#   'all'       — 三模态全融合
#   'rgb_ir'    — RGB + IR (depth 置零)
#   'rgb_depth' — RGB + Depth (ir 置零)
#   'rgb_only'  — 纯 RGB (跳过 IR/Depth 编码和融合)
# ============================================================
fusion_modality = 'all'

# ============================================================
# 辅助损失权重 (Scene Query)
# ============================================================
scene_loss_coef = 0.1           # 光照/密度分类损失权重
weight_entropy_coef = 0.01      # 模态权重熵正则系数

# ============================================================
# 优化器 & 显存
# ============================================================
batch_size = 2                  # V100 16GB 极限
weight_decay = 1e-4
clip_max_norm = 0.1             # 梯度裁剪
use_ema = True                  # 启用 EMA
ema_decay = 0.9999
ema_epoch = 10                  # 第 10 epoch 开始 EMA

# ============================================================
# 预训练权重加载 (可选)
# ============================================================
# 若指定 pretrain_model_path, 加载 COCO 预训练的 DINO 权重
# 并忽略 label_enc.weight 和 class_embed (类别数不同)
# 命令行: --pretrain_model_path /path/to/checkpoint.pth
#         --finetune_ignore label_enc.weight class_embed
