#!/usr/bin/env python3
"""Quick RefCOCO evaluation - load best checkpoint, evaluate on RefCOCO val."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import datasets.transforms as T
from datasets.odvg import ODVGDataset
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

def main():
    # Load model
    cfg = SLConfig.fromfile('config/cfg_refcoco_swinb.py')
    args = type('Args', (), {})()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    args.device = 'cuda'
    args.coco_val_path = '/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/annotations/instances_val2014_5k.json'

    build_func = MODULE_BUILD_FUNCS.get('groundingdino')
    model, _, _ = build_func(args)
    ckpt = torch.load('output/refcoco_all_swinb_baseline/checkpoint_best_regular.pth', map_location='cpu')
    model.load_state_dict(clean_state_dict(ckpt['model']), strict=False)
    model = model.cuda().eval()
    print('Model loaded')

    # Load RefCOCO val
    ds = ODVGDataset(
        root='/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/train2014',
        anno='/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/refcoco/refcoco_val_odvg.jsonl',
    )
    print(f'Val images: {len(ds)}')

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

        for region in regions:
            gt_bbox = region['bbox']  # [x1,y1,x2,y2] in pixels
            query = region['phrase'].lower().strip() + '.'
            gt_norm = [gt_bbox[0]/img_w, gt_bbox[1]/img_h, gt_bbox[2]/img_w, gt_bbox[3]/img_h]

            img_tensor, _ = transform(image_pil, None)
            img_tensor = img_tensor.cuda()
            with torch.no_grad():
                outputs = model(img_tensor[None], captions=[query])

            logits = outputs['pred_logits'].sigmoid()[0]
            boxes = outputs['pred_boxes'][0]  # cxcywh normalized
            filt = logits.max(dim=1)[0] > 0.3
            boxes_filt = boxes[filt]
            scores_filt = logits[filt].max(dim=1)[0]

            if len(boxes_filt) > 0:
                best = scores_filt.argmax()
                cx, cy, w, h = boxes_filt[best].tolist()
                pred = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                px1,py1,px2,py2 = pred
                gx1,gy1,gx2,gy2 = gt_norm
                xi1,yi1 = max(px1,gx1), max(py1,gy1)
                xi2,yi2 = min(px2,gx2), min(py2,gy2)
                inter = max(0, xi2-xi1) * max(0, yi2-yi1)
                a1 = (px2-px1)*(py2-py1)
                a2 = (gx2-gx1)*(gy2-gy1)
                iou = inter / (a1 + a2 - inter + 1e-8)
            else:
                iou = 0.0
            ious.append(iou)

        if (idx+1) % 200 == 0:
            acc05 = sum(1 for i in ious if i >= 0.5) / len(ious)
            print(f'  [{idx+1}/{len(ds)}] samples={len(ious)} acc@0.5={acc05:.4f}')

    acc05 = sum(1 for i in ious if i >= 0.5) / len(ious)
    acc07 = sum(1 for i in ious if i >= 0.7) / len(ious)
    mean_iou = sum(ious) / len(ious)
    print(f'\n=== RefCOCO Val Results ===')
    print(f'Total samples: {len(ious)}')
    print(f'Acc@0.5:  {acc05:.4f}')
    print(f'Acc@0.7:  {acc07:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')

if __name__ == '__main__':
    main()
