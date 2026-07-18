# Competition Adaptation Plan: Multi-Modal Visual Grounding

## Competition Overview

- **Task**: Given RGB + Infrared + Depth images + English text query, output bbox on visible image
- **Metric**: ACC@0.5 (percentage of predictions with IoU >= 0.5)
- **Data**: 2000 images, 9555 queries, test set only (no bbox in JSON)

---

## Hardware

| GPU | Memory | Count | Total |
|-----|--------|-------|-------|
| Tesla PG503-216 (V100 32GB) | 32 GB | 8 | 256 GB |

---

## 1. Dataset Structure (原始数据格式)

### 1.1 数据目录结构
```
/home/zlab8v100/ssd4/bigdata4/aic2026/
├── Images/
│   ├── visible/      # 三通道 RGB, 8-bit, [0,255]
│   ├── depth/        # 单通道, 16-bit, 单位 mm, 范围约 [0,19999]
│   └── infrared/     # 三通道 (实为灰度堆叠), 8-bit, [0,255]
└── queries/
    └── queries.json  # 9555 条 query, 无 bbox (测试集)
```

### 1.2 文件命名规则
- 所有模态图像使用相同文件名，如 `000002.png`
- 路径: `Images/{visible,infrared,depth}/000002.png`
- 共 2000 张图，每张图有 1-7 个 query

### 1.3 queries.json 格式 (输入)
```json
{
  "000002_001": {
    "visible": "Images/visible/000002.png",
    "infrared": "Images/infrared/000002.png",
    "depth": "Images/depth/000002.png",
    "query": "Red promotional sign with food imagery"
  },
  ...
}
```
- Key: `{6-digit-image-id}_{3-digit-query-num}`
- 测试集中**没有** `bbox` 字段

### 1.4 输出格式（竞赛要求）
```json
{
  "000002_001": {
    "visible": "Images/visible/000002.png",
    "infrared": "Images/infrared/000002.png",
    "depth": "Images/depth/000002.png",
    "query": "Red promotional sign with food imagery",
    "bbox": [x1, y1, x2, y2]   // 归一化 [0,1], 基于 visible 图像
  },
  ...
}
```
- 所有字段不可修改，仅添加 `bbox`
- 最终提交为 zip 压缩包
- bbox 坐标必须有效（x1<x2, y1<y2, 不越界, 非 NaN）

---

## 2. 输入数据管线改造

### 2.1 新数据集类: `AicMultiModalDataset`

需要新建 `datasets/aic_dataset.py`:

```python
class AicMultiModalDataset(Dataset):
    """
    读取赛题 JSON, 返回四模态数据 + target bbox (如果有)
    """
    def __init__(self, json_path, image_root, transforms=None, is_train=True):
        # 解析 queries.json
        # 为每张图缓存三模态路径，避免重复读取
        pass

    def __getitem__(self, idx):
        return {
            "visible": Tensor[3, H, W],    # 归一化后的 RGB
            "depth":   Tensor[1, H, W],    # 单通道深度，已处理无效值
            "infrared": Tensor[1, H, W],   # 红外取第一个通道（三通道相同）
            "query":   str,                # 文本查询
            "query_id": str,               # "000002_001"
            "bbox":    Tensor[4] or None,  # 归一化 [x1,y1,x2,y2]
            "img_size": (h, w, orig_w, orig_h)
        }
```

### 2.2 各模态预处理规则

| 模态 | 原始格式 | 预处理 |
|------|----------|--------|
| **visible** | 3-ch RGB, uint8, [0,255] | resize + normalize (ImageNet mean/std) + 转 float32 |
| **infrared** | 3-ch uint8, [0,255], 三通道相同 | 取第1通道 → [1,H,W]，resize + normalize，转 float32 |
| **depth** | 1-ch uint16, mm, [0,~20000] | clip(0, 20000) → /20000 normalize 到 [0,1] → resize，转 float32；值为0的区域设为 mask |

### 2.3 三模态空间对齐
- 三个模态的图像尺寸已对齐（相同分辨率），直接 channel-concat 或独立编码均可

---

## 3. 模型改造方案

### 3.1 设计约束 (关键)

1. **模态可选开关**: 三个视觉通道 (visible/depth/infrared) 必须通过 **arg 参数** 来独立控制开启/关闭
2. **保留官方原模型作为 baseline**: 原版 GroundingDINO 代码和权重必须可用，用于对比实验
3. **预训练权重兼容**: 改造后的模型需尽可能复用预训练权重

### 3.2 核心架构：渐进式交叉注意力级联 (Progressive Cross-Attention Cascade)

**设计思路** (参考时序预测领域 Cross-Attention 模式):
- 文本 → BERT → 转化为 Query Q
- Q 依次与三个模态的视觉特征做交叉注意力，逐步累积多模态信息
- Q → CA(Q, RGB) → Q' → CA(Q', Infrared) → Q'' → CA(Q'', Depth) → Q''' → bbox

```
                          文本 Query
                              │
                     BERT Text Encoder
                              │
                    Linear Proj → Q [B, N_q, 256]
                              │
          ┌───────────────────┤
          │                   │
          ▼                   ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Visible      │   │ Infrared     │   │ Depth        │
   │ Swin-T (3ch) │   │ Swin-T (1ch) │   │ Swin-T (1ch) │
   │ PRETRAINED   │   │ scratch      │   │ scratch      │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │                  │                  │
          ▼                  ▼                  ▼
   vis_multiscale      ir_multiscale      dp_multiscale
   features [4 scales] features [4 scales] features [4 scales]
          │                  │                  │
          └──────┬───────────┴─────────┬────────┘
                 │                     │
                 ▼                     │
   Step 1: CrossAttn(Q, vis_feat)      │
   Q ← Q + CA(Q, vis_feat)             │
   Q ← LayerNorm(Q)                    │
   Q ← Q + FFN(Q)                      │
                 │                     │
                 ▼                     │
   Step 2: CrossAttn(Q, ir_feat)       │
   Q ← Q + CA(Q, ir_feat)              │
   Q ← LayerNorm(Q)                    │
   Q ← Q + FFN(Q)                      │
                 │                     │
                 ▼                     ▼
   Step 3: CrossAttn(Q, dp_feat)
   Q ← Q + CA(Q, dp_feat)
   Q ← LayerNorm(Q)
   Q ← Q + FFN(Q)
                 │
                 ▼
          ┌──────────────┐
          │  BBox Head   │
          │ MLP(256→256→4)│
          └──────┬───────┘
                 │
                 ▼
        bbox [B, N_q, 4] 归一化 (cx,cy,w,h)
```

> **为什么用这个顺序 (RGB → IR → Depth)?**
> RGB 提供外观纹理信息(最丰富), IR 提供热源/活体信息(区分人与背景), Depth 提供空间几何信息(验证距离/尺寸)。Q 先获取外观语义，再补充热特征，最后用空间几何做精确约束——符合"从粗到细"的信息累积逻辑。

### 3.3 各组件设计详解

#### 3.3.1 三模态 Backbone (输出结构一致)

原始 RGB 的 Swin-T 输出结构 (来自 `config/cfg_odvg.py`):
```
backbone.num_channels = [192, 384, 768]  # 3 个尺度
num_feature_levels = 4                    # 额外下采样一层 → 共 4 个尺度
input_proj: Conv1x1 → 每个尺度投影到 hidden_dim=256
```

改造后，每个模态独立 backbone，输出结构**完全一致**:

```python
# 三份 backbone 实例，输出同一组多尺度特征
vis_backbone = SwinTransformer(in_chans=3, ...)   # 复用预训练权重
ir_backbone  = SwinTransformer(in_chans=1, ...)   # 仅第一层 patch_embed 不同
dp_backbone  = SwinTransformer(in_chans=1, ...)   # 仅第一层 patch_embed 不同

# 每个 backbone 输出: List[NestedTensor]  length=3 (多尺度)
# scales: stride8, stride16, stride32
# channels per scale: [192, 384, 768]

# 每个模态独立 input_proj, 投影到 hidden_dim=256
vis_srcs = [self.input_proj_vis[i](f.decompose()[0]) for i, f in enumerate(vis_features)]
ir_srcs  = [self.input_proj_ir[i](f.decompose()[0])  for i, f in enumerate(ir_features)]
dp_srcs  = [self.input_proj_dp[i](f.decompose()[0])  for i, f in enumerate(dp_features)]

# 结果: 每个模态 [B, 256, H_i, W_i] × 4 scales
```

#### 3.3.2 多尺度特征处理方式（对齐原模型）

**原模型 (`transformer.py:222-249`): 所有尺度全部 flatten 拼接**

```python
# 原模型 encoder 入口: 每个尺度的空间位置全部展开拼接
src_flatten = []
mask_flatten = []
lvl_pos_embed_flatten = []
spatial_shapes = []

for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
    bs, c, h, w = src.shape
    spatial_shapes.append((h, w))
    src = src.flatten(2).transpose(1, 2)           # [B,C,H,W] → [B, H*W, C]
    mask = mask.flatten(1)                          # [B, H, W] → [B, H*W]
    pos_embed = pos_embed.flatten(2).transpose(1, 2) # [B, C, H, W] → [B, H*W, C]
    # 加 level_embed 区分尺度
    lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
    lvl_pos_embed_flatten.append(lvl_pos_embed)
    src_flatten.append(src)
    mask_flatten.append(mask)

# 拼接所有尺度 → 得到一个巨大的 KV 集合
src_flatten = torch.cat(src_flatten, 1)             # [B, Σ(H_i*W_i), 256]
lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
spatial_shapes = torch.as_tensor(spatial_shapes)    # [[H0,W0], [H1,W1], [H2,W2], [H3,W3]]
level_start_index = torch.cat((zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]))
```

**我们的 Cross-Attention 采用同样策略**:

```python
def flatten_multiscale_features(srcs, masks, pos_embeds, level_embed):
    """
    将多尺度特征展平为 [B, Σ(H_i*W_i), 256] 的 token 序列
    srcs: List of [B, 256, H_i, W_i]
    masks: List of [B, H_i, W_i]
    pos_embeds: List of [B, 256, H_i, W_i]
    """
    src_flatten = []
    pos_flatten = []
    mask_flatten = []
    for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
        src = src.flatten(2).transpose(1, 2)          # [B, H_i*W_i, 256]
        mask = mask.flatten(1)                        # [B, H_i*W_i]
        pos = pos.flatten(2).transpose(1, 2)          # [B, H_i*W_i, 256]
        pos = pos + level_embed[lvl].view(1, 1, -1)   # add scale identifier
        src_flatten.append(src)
        pos_flatten.append(pos)
        mask_flatten.append(mask)
    src_flatten = torch.cat(src_flatten, 1)           # [B, Σ(H_i*W_i), 256]
    pos_flatten = torch.cat(pos_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)         # [B, Σ(H_i*W_i)]
    return src_flatten, pos_flatten, mask_flatten

# 三个模态各自 flatten → 分别得到三个 KV 集合
vis_kv, vis_pos, vis_mask = flatten_multiscale_features(vis_srcs, vis_masks, vis_poss, self.level_embed_vis)
ir_kv,  ir_pos,  ir_mask  = flatten_multiscale_features(ir_srcs,  ir_masks,  ir_poss,  self.level_embed_ir)
dp_kv,  dp_pos,  dp_mask  = flatten_multiscale_features(dp_srcs,  dp_masks,  dp_poss,  self.level_embed_dp)
```

> **关键点**:
> - 不丢失任何尺度信息，模型在 Cross-Attention 时可以自行学习关注哪个尺度
> - `level_embed` 是每个尺度独立的可学习向量，让模型区分 token 来自哪个尺度
> - 输入 800×800 图像时，4 个尺度 token 数约: 100×100 + 50×50 + 25×25 + 13×13 ≈ 13,600 tokens
> - 3 个模态 × 3 个 CA block = 每个 batch 做 3 次 Cross-Attention(Q=900, KV=13,600)，显存可控
```

#### 3.3.2 文本 → Q 的转化

```python
# 沿用原 BERT 编码 + feat_map
bert_output = self.bert(**tokenized)        # [B, L, 768]
text_feat = self.feat_map(bert_output)      # [B, L, 256]

# 池化为 Query Q
# 方案: 取 CLS token + Mean pool → concat → 扩展为 N_q 个 query
cls_feat = text_feat[:, 0, :]               # [B, 256]
sent_feat = text_feat[:, 1:, :].mean(dim=1) # [B, 256]
q_feat = cls_feat + sent_feat               # [B, 256]

# 可学习 query 初始化为文本特征的扩展
self.query_embed = nn.Embedding(num_queries, 256)  # 可学习，或通过 q_feat 初始化
Q = q_feat.unsqueeze(1).expand(-1, num_queries, -1)  # [B, N_q, 256]
Q = Q + self.query_embed.weight.unsqueeze(0)          # [B, N_q, 256]
```

#### 3.3.3 渐进式 Cross-Attention 模块

每个 CA Step 的结构 (参考 Transformer Decoder Layer):

```python
class ProgressiveCrossAttentionBlock(nn.Module):
    """一个 Cross-Attention step: Q ← Q + CA(Q, modality_feat)"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, Q, feat_kv, feat_mask=None):
        # feat_kv: [B, N_tokens, 256] (某模态所有尺度展平后的特征)
        # Q: [B, N_q, 256]
        Q2 = self.cross_attn(Q, feat_kv, feat_kv, key_padding_mask=feat_mask)[0]
        Q = self.norm1(Q + Q2)          # residual + norm
        Q = self.norm2(Q + self.ffn(Q)) # FFN + residual + norm
        return Q
```

**三级联调用**:
```python
class ProgressiveCrossAttentionCascade(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        self.ca_vis = ProgressiveCrossAttentionBlock(d_model, nhead)
        self.ca_ir  = ProgressiveCrossAttentionBlock(d_model, nhead)
        self.ca_dp  = ProgressiveCrossAttentionBlock(d_model, nhead)

    def forward(self, Q, vis_feat, ir_feat, dp_feat,
                use_vis=True, use_ir=True, use_dp=True):
        """
        vis_feat, ir_feat, dp_feat: [B, N_tokens, 256]
        Q: [B, N_q, 256]
        """
        if use_vis:
            Q = self.ca_vis(Q, vis_feat)
        if use_ir:
            Q = self.ca_ir(Q, ir_feat)
        if use_dp:
            Q = self.ca_dp(Q, dp_feat)
        return Q
```

#### 3.3.4 检测头

```python
# 从 Q 预测 bbox (保留原模型风格)
# Q: [B, N_q, 256] → MLP → [B, N_q, 4]
self.bbox_head = MLP(256, 256, 4, 3)  # 3层 MLP

# 取 top-1 预测 (因为每个 query 只描述一个目标)
bbox = self.bbox_head(Q)  # [B, N_q, 4]
# 选取 CLS token 对应的预测，或 score 最高的
```

### 3.4 预训练权重兼容性

| 组件 | 来源 | 复用策略 |
|------|------|----------|
| **vis_backbone** (Swin-T, 3ch) | 原 GroundingDINO 权重 | 完整加载 |
| **ir_backbone** (Swin-T, 1ch) | 无预训练 | 第一层随机初始化，其余层从 vis_backbone 复制第2层后的权重 |
| **dp_backbone** (Swin-T, 1ch) | 无预训练 | 同上 |
| **BERT** | 原权重 | 完整加载 |
| **feat_map** (Linear) | 原权重 | 完整加载 |
| **CA blocks** (3个) | 新建 | 随机初始化 |
| **bbox_head** | 新建 | 随机初始化 |

> **关键技术**: ir_backbone 和 dp_backbone 的第一层 `PatchEmbed.proj` 是 `Conv2d(1, 96, 4)` (vs 原 `Conv2d(3, 96, 4)`)，但后续所有层结构完全相同。
> 可以将 vis_backbone 的 PatchEmbed 权重对 RGB 三通道取平均作为单通道初始化:
> ```python
> # 将 3ch conv 权重平均为 1ch 权重
> ir_backbone.patch_embed.proj.weight.data = vis_backbone.patch_embed.proj.weight.data.mean(dim=1, keepdim=True)
> # 后续 Stage 权重直接复制
> ir_backbone.layers.load_state_dict(vis_backbone.layers.state_dict())
> ```

### 3.5 开关参数设计

```python
# 模型初始化参数
model = GroundingDINOProgressiveCA(
    use_depth=True,           # 是否启用深度模态
    use_infrared=True,        # 是否启用红外模态
    vis_backbone='swin_T_224_1k',
    ir_backbone='swin_T_224_1k',
    dp_backbone='swin_T_224_1k',
    num_queries=1,            # 每 query 预测一个目标 → bbox
    d_model=256,
    nhead=8,
    pretrained_weights='path/to/groundingdino.pth',
)
```

**关键行为**:
- `use_depth=False, use_infrared=False`: 只有 RGB → CA(Q, vis_feat) → bbox，最简 baseline
- `use_depth=False, use_infrared=True`: RGB + IR 双模态
- `use_depth=True, use_infrared=True`: 全三模态

---

## 4. 输出格式改造

### 4.1 预测输出函数
新建或修改推理脚本，读取 queries.json，遍历所有 query，每个输出 bbox:

```python
def predict_and_export(model, queries_json_path, image_root, output_path):
    # 1. 读取 queries.json
    # 2. 逐条推理
    # 3. 将 bbox 写入原 JSON 结构
    # 4. 输出为 result.json
    # 5. 打包为 zip
```

### 4.2 输出校验
- bbox: `[x1, y1, x2, y2]` 归一化 [0,1]
- 检查 x1<x2, y1<y2
- 检查不越界
- 检查无 NaN
- 非法框设置为全图框 [0, 0, 1, 1] (兜底策略)

---

## 5. 评测指标改造

### 5.1 当前指标: COCO mAP
- AP@[0.5:0.95], AP@0.5, AP@0.75, AP_small/medium/large
- 使用 pycocotools

### 5.2 竞赛指标: ACC@0.5
```python
def compute_acc_at_05(pred_boxes, gt_boxes):
    """
    pred_boxes: dict {query_id: [x1,y1,x2,y2]}
    gt_boxes:   dict {query_id: [x1,y1,x2,y2]}
    Returns: percentage of predictions with IoU >= 0.5
    """
    correct = 0
    total = len(pred_boxes)
    for qid in pred_boxes:
        if qid in gt_boxes:
            iou = box_iou(pred_boxes[qid], gt_boxes[qid])
            if iou >= 0.5:
                correct += 1
    return correct / total
```

---

## 6. 关键文件改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `datasets/aic_dataset.py` | **新建** | 赛题数据集类，读取四模态 |
| `config/cfg_aic.py` | **新建** | 赛题训练配置文件 |
| `tools/inference_aic.py` | **新建** | 赛题推理 + 输出 JSON + zip |
| `models/groundingdino_mm.py` | **新建** | 多模态 GroundingDINO（添加 depth/infrared encoder） |
| `main_mm.py` | **新建** | 多模态训练入口 |
| `util/metrics.py` | **新建** | ACC@0.5 评测函数 |
| `models/GroundingDINO/groundingdino.py` | 不改 | 保留原模型作为 baseline |

---

## 7. 待确认问题

1. **训练数据**: 是否有带 bbox 标注的训练集？（用户自己去找）
2. **预训练权重**: 官方 GroundingDINO pretrained weights 路径？
3. ~~计算资源~~ → 8x V100 32GB，充足
