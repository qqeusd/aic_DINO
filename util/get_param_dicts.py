import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd', 'multimodal']

    # by default
    if param_dict_type == 'default':
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
        return param_dicts

    if param_dict_type == 'ddetr_in_mmdet':
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]        
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]

        # print("param_dicts: {}".format(param_dicts))

    return param_dicts


# [新增] 三模态分层学习率策略
def get_multimodal_param_dict(args, model_without_ddp: nn.Module):
    """
    三模态微调的分层学习率配置。

    分层策略:
        组1 — 新模块 (高学习率): ir_encoder, depth_encoder, scene_fusion
        组2 — Transformer 微调 (低学习率): transformer, bbox_embed, class_embed,
              input_proj, label_enc, tgt_embed, level_embed
        组3 — Backbone: 由 requires_grad 控制 (freeze_backbone 时自动排除)

    Args:
        args: 全局配置，需包含:
            - lr: 新模块学习率 (e.g. 1e-4)
            - lr_multimodal_transformer: Transformer 微调学习率 (e.g. 1e-5)
        model_without_ddp: DINO 模型 (不含 DDP wrapper)

    Returns:
        List[dict]: param_dicts，供 optimizer 使用。
    """
    lr_new = args.lr
    lr_transformer = getattr(args, 'lr_multimodal_transformer', 1e-5)

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, ['ir_encoder', 'depth_encoder', 'scene_fusion'])
                       and p.requires_grad],
            "lr": lr_new,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if not match_name_keywords(n, ['ir_encoder', 'depth_encoder', 'scene_fusion',
                                                       'backbone'])
                       and p.requires_grad],
            "lr": lr_transformer,
        },
    ]

    # 过滤掉空 param group
    param_dicts = [pg for pg in param_dicts if len(pg["params"]) > 0]

    return param_dicts