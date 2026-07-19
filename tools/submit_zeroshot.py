#!/usr/bin/env python3
"""
Zero-shot inference on competition test set using pretrained GroundingDINO.
Generates submission JSON with predictions for all 9555 queries.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm
import datasets.transforms as T
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def main():
    # Paths
    queries_json = '/home/zlab8v100/ssd4/bigdata4/aic2026/queries/queries.json'
    image_root = '/home/zlab8v100/ssd4/bigdata4/aic2026/'
    pretrained_path = '/home/zcoop8/zhangxianping/groundingdino/pretrained/groundingdino_swinb_cogcoor.pth'
    output_dir = 'output/submission_zeroshot'
    os.makedirs(output_dir, exist_ok=True)

    # Load queries
    with open(queries_json) as f:
        queries = json.load(f)
    print(f'Total queries: {len(queries)}')

    # Load model (zero-shot, no fine-tuning)
    cfg = SLConfig.fromfile('config/cfg_refcoco_swinb.py')
    args = type('Args', (), {})()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    args.device = 'cuda'
    args.coco_val_path = '/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/annotations/instances_val2014_5k.json'
    build_func = MODULE_BUILD_FUNCS.get('groundingdino')
    model, _, _ = build_func(args)

    ckpt = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(ckpt['model']), strict=False)
    model = model.cuda().eval()
    print('Model loaded (zero-shot)')

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = {}
    box_threshold = 0.15
    invalid_count = 0

    for key, q in tqdm(queries.items(), desc='Inference'):
        img_path = os.path.join(image_root, q['visible'])
        image_pil = Image.open(img_path).convert('RGB')
        img_w, img_h = image_pil.size
        query_text = q['query'].lower().strip() + '.'

        img_tensor, _ = transform(image_pil, None)
        img_tensor = img_tensor.cuda()

        with torch.no_grad():
            outputs = model(img_tensor[None], captions=[query_text])

        logits = outputs['pred_logits'].sigmoid()[0]
        boxes = outputs['pred_boxes'][0]
        scores = logits.max(dim=1)[0]
        top_idx = scores.argmax()
        top_score = scores[top_idx].item()

        if top_score < box_threshold:
            # No valid prediction, return full-image box
            bbox = [0.0, 0.0, 1.0, 1.0]
            invalid_count += 1
        else:
            cx, cy, w, h = boxes[top_idx].tolist()
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            # Validate
            if x1 >= x2 or y1 >= y2:
                bbox = [0.0, 0.0, 1.0, 1.0]
                invalid_count += 1
            else:
                bbox = [x1, y1, x2, y2]

        results[key] = {
            "visible": q['visible'],
            "infrared": q['infrared'],
            "depth": q['depth'],
            "query": q['query'],
            "bbox": bbox,
        }

    # Save results
    output_json = os.path.join(output_dir, 'results.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {len(results)} predictions to {output_json}')
    print(f'Invalid/low-conf predictions (fallback to [0,0,1,1]): {invalid_count}')

    # Create zip
    import zipfile
    zip_path = os.path.join(output_dir, 'submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, 'results.json')
    print(f'Submission zip: {zip_path}')


if __name__ == '__main__':
    main()
