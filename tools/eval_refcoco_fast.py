#!/usr/bin/env python3
"""Fast batch RefCOCO evaluation - batched queries per image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, torch
from PIL import Image
from collections import defaultdict
sys.path.insert(0, '.')
import datasets.transforms as T
from datasets.odvg import ODVGDataset
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

def evaluate_checkpoint(ckpt_path, ds, desc, device='cuda:0'):
    cfg = SLConfig.fromfile('config/cfg_refcoco_swinb.py')
    args = type('Args', (), {})()
    for k, v in cfg._cfg_dict.to_dict().items(): setattr(args, k, v)
    args.device = device
    args.coco_val_path = '/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/annotations/instances_val2014_5k.json'
    build_func = MODULE_BUILD_FUNCS.get('groundingdino')
    model, _, _ = build_func(args)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(ckpt['model']), strict=False)
    model = model.to(device).eval()

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ious = []
    for idx in range(len(ds)):
        meta = ds.metas[idx]
        img_path = os.path.join(ds.root, meta['filename'])
        image_pil = Image.open(img_path).convert('RGB')
        img_w, img_h = image_pil.size
        regions = meta['grounding']['regions']

        # Batch all queries for this image
        queries = [r['phrase'].lower().strip() + '.' for r in regions]
        gt_boxes_norm = [[r['bbox'][0]/img_w, r['bbox'][1]/img_h, r['bbox'][2]/img_w, r['bbox'][3]/img_h] for r in regions]

        img_tensor, _ = transform(image_pil, None)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor[None], captions=queries)

        logits_all = outputs['pred_logits'].sigmoid()[0]  # [n_phrases*nq, 256]
        boxes_all = outputs['pred_boxes'][0]              # [n_phrases*nq, 4]
        n_phrases = len(queries)
        nq_per = 900

        for qi in range(n_phrases):
            start, end = qi * nq_per, (qi + 1) * nq_per
            logits = logits_all[start:end]
            boxes = boxes_all[start:end]
            scores = logits.max(dim=1)[0]
            best = scores.argmax()
            b = boxes[best].tolist()
            cx, cy, w, h = b
            px1 = cx - w/2; py1 = cy - h/2
            px2 = cx + w/2; py2 = cy + h/2
            gx1, gy1, gx2, gy2 = gt_boxes_norm[qi]
            xi1, yi1 = max(px1, gx1), max(py1, gy1)
            xi2, yi2 = min(px2, gx2), min(py2, gy2)
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            a1 = (px2-px1)*(py2-py1); a2 = (gx2-gx1)*(gy2-gy1)
            iou = inter / (a1 + a2 - inter + 1e-8)
            ious.append(iou)

        if (idx+1) % 300 == 0:
            acc05 = sum(1 for i in ious if i >= 0.5) / len(ious)
            print(f'  [{desc}] {idx+1}/{len(ds)} acc@0.5={acc05:.4f}')

    acc05 = sum(1 for i in ious if i >= 0.5) / len(ious)
    acc07 = sum(1 for i in ious if i >= 0.7) / len(ious)
    mean_iou = sum(ious) / len(ious)
    print(f'\n=== [{desc}] Results ===')
    print(f'Total: {len(ious)}  Acc@0.5: {acc05:.4f}  Acc@0.7: {acc07:.4f}  Mean IoU: {mean_iou:.4f}')
    return {'acc05': acc05, 'acc07': acc07, 'mean_iou': mean_iou}

if __name__ == '__main__':
    device = f'cuda:{sys.argv[1]}' if len(sys.argv) > 1 else 'cuda:0'
    ckpt = sys.argv[2] if len(sys.argv) > 2 else 'output/refcoco_all_swinb_baseline/checkpoint_best_regular.pth'
    desc = sys.argv[3] if len(sys.argv) > 3 else 'eval'

    ds = ODVGDataset(
        root='/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/train2014',
        anno='/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/refcoco/refcoco_val_odvg.jsonl',
    )
    print(f'Val: {len(ds)} images, evaluating on {device}')
    evaluate_checkpoint(ckpt, ds, desc, device)
