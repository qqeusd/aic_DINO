"""
Evaluation script for RefCOCO/RefCOCO+/RefCOCOg visual grounding.

Computes Precision@IoU (Acc@0.5) and visualizes predictions vs ground truth.

Usage:
  python tools/eval_refcoco.py \
    --config_file config/cfg_refcoco.py \
    --checkpoint_path output/checkpoint.pth \
    --datasets config/datasets_refcoco.json \
    --output_dir output/eval_results \
    --visualize 10
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.transforms as T
from datasets.odvg import ODVGDataset
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def box_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_acc(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compute accuracy at IoU threshold.

    Args:
        pred_boxes: list of predicted boxes [x1,y1,x2,y2] (absolute coords)
        gt_boxes: list of ground truth boxes [x1,y1,x2,y2] (absolute coords)

    Returns:
        acc: float, accuracy
        ious: list of IoU values
    """
    correct = 0
    ious = []
    for pb, gb in zip(pred_boxes, gt_boxes):
        iou = box_iou(pb, gb)
        ious.append(iou)
        if iou >= iou_threshold:
            correct += 1
    acc = correct / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    return acc, ious


def load_model(config_path, checkpoint_path, device='cuda'):
    args = SLConfig.fromfile(config_path)
    args.device = device
    # Set necessary attributes for build (required by PostProcess)
    args.coco_val_path = '/home/zlab8v100/ssd4/bigdata4/aic2026/coco2014/annotations/instances_val2014_5k.json'

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.to(device)
    model.eval()
    return model


def get_prediction(model, image_tensor, caption, device='cuda', box_threshold=0.3, text_threshold=0.25):
    """Run inference on a single image + caption pair."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs['pred_logits'].sigmoid()[0]    # (nq, 256)
    boxes = outputs['pred_boxes'][0]                 # (nq, 4)  cxcywh format

    # Filter by box score
    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]
    scores_filt = logits[filt_mask].max(dim=1)[0]

    if len(boxes_filt) == 0:
        return None, 0.0

    # Pick the highest-scoring box
    best_idx = scores_filt.argmax()
    best_box = boxes_filt[best_idx]
    best_score = scores_filt[best_idx].item()

    # Convert cxcywh to xyxy (both normalized 0-1)
    cx, cy, w, h = best_box.tolist()
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return [x1, y1, x2, y2], best_score


def denorm_boxes(pred_box_norm, gt_box_norm, img_w, img_h):
    """Convert normalized boxes to absolute pixel coordinates."""
    p_x1, p_y1, p_x2, p_y2 = pred_box_norm
    pred_abs = [p_x1 * img_w, p_y1 * img_h, p_x2 * img_w, p_y2 * img_h]

    g_x1, g_y1, g_x2, g_y2 = gt_box_norm
    gt_abs = [g_x1 * img_w, g_y1 * img_h, g_x2 * img_w, g_y2 * img_h]

    return pred_abs, gt_abs


def draw_boxes_on_image(image_pil, pred_box, gt_box, query_text, save_path):
    """
    Draw prediction (green) and ground truth (red) boxes on image.

    Args:
        image_pil: PIL Image
        pred_box: [x1, y1, x2, y2] absolute pixel coords
        gt_box: [x1, y1, x2, y2] absolute pixel coords
        query_text: text query
        save_path: path to save the visualized image
    """
    draw = ImageDraw.Draw(image_pil)

    # Draw GT box (red)
    if gt_box is not None:
        gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
        draw.rectangle([gx1, gy1, gx2, gy2], outline='red', width=4)
        draw.text((gx1, max(0, gy1 - 20)), f"GT: {query_text[:40]}", fill='red')

    # Draw Pred box (green)
    if pred_box is not None:
        px1, py1, px2, py2 = [int(v) for v in pred_box]
        draw.rectangle([px1, py1, px2, py2], outline='lime', width=4)
        draw.text((px1, py2 + 2), "PRED", fill='lime')

    image_pil.save(save_path)


def evaluate(model, val_dataset, device='cuda', visualize_dir=None, max_vis=20):
    """Run evaluation on a validation dataset."""
    all_pred_boxes = []
    all_gt_boxes = []
    all_ious = []
    vis_count = 0

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for idx in range(len(val_dataset)):
        meta = val_dataset.metas[idx]

        # Load image
        img_path = os.path.join(val_dataset.root, meta['filename'])
        image_pil = Image.open(img_path).convert('RGB')
        img_w, img_h = image_pil.size

        anno = meta['grounding']
        regions = anno['regions']

        for region in regions:
            gt_bbox_norm = region['bbox']  # [x1, y1, x2, y2] in pixel coords
            query = region['phrase']

            # Normalize GT bbox
            gt_box_norm = [
                gt_bbox_norm[0] / img_w,
                gt_bbox_norm[1] / img_h,
                gt_bbox_norm[2] / img_w,
                gt_bbox_norm[3] / img_h,
            ]

            # Run inference
            img_tensor, _ = transform(image_pil, None)
            pred_box_norm, score = get_prediction(model, img_tensor, query, device)

            if pred_box_norm is not None:
                pred_abs, gt_abs = denorm_boxes(pred_box_norm, gt_box_norm, img_w, img_h)
                iou = box_iou(pred_abs, gt_abs)
            else:
                # No prediction -> IoU = 0
                pred_abs = None
                gt_abs = [
                    gt_box_norm[0] * img_w, gt_box_norm[1] * img_h,
                    gt_box_norm[2] * img_w, gt_box_norm[3] * img_h,
                ]
                iou = 0.0

            all_ious.append(iou)
            all_pred_boxes.append(pred_abs)
            all_gt_boxes.append(gt_abs)

            # Visualize
            if visualize_dir is not None and vis_count < max_vis:
                vis_path = os.path.join(visualize_dir, f"vis_{vis_count:04d}.jpg")
                draw_boxes_on_image(image_pil.copy(), pred_abs, gt_abs, query, vis_path)
                vis_count += 1

    # Compute metrics
    acc_05 = sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious)
    acc_07 = sum(1 for iou in all_ious if iou >= 0.7) / len(all_ious)
    mean_iou = np.mean(all_ious)

    return {
        'acc@0.5': acc_05,
        'acc@0.7': acc_07,
        'mean_iou': mean_iou,
        'num_samples': len(all_ious),
        'ious': all_ious,
    }


def main():
    parser = argparse.ArgumentParser("RefCOCO Evaluation with Visualization")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--datasets", "-d", type=str, required=True,
                        help="Path to dataset JSON config")
    parser.add_argument("--output_dir", "-o", type=str, default="output/eval_refcoco")
    parser.add_argument("--visualize", "-v", type=int, default=20,
                        help="Number of samples to visualize")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = 'cpu' if args.cpu_only else 'cuda'

    # Load datasets config
    with open(args.datasets, 'r') as f:
        dataset_config = json.load(f)

    # Load model
    print("Loading model...")
    model = load_model(args.config_file, args.checkpoint_path, device)
    print("Model loaded.")

    # Evaluate on each val dataset
    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = {}

    for ds_info in dataset_config.get('val', []):
        ds_name = os.path.basename(ds_info['anno']).replace('_odvg.jsonl', '')
        print(f"\nEvaluating on: {ds_name}")

        val_dataset = ODVGDataset(
            root=ds_info['root'],
            anno=ds_info['anno'],
        )
        print(f"  == {len(val_dataset)} images")

        vis_dir = os.path.join(args.output_dir, f"vis_{ds_name}")
        os.makedirs(vis_dir, exist_ok=True)

        metrics = evaluate(
            model, val_dataset, device,
            visualize_dir=vis_dir,
            max_vis=args.visualize,
        )
        all_metrics[ds_name] = {k: v for k, v in metrics.items() if k != 'ious'}

        print(f"  Acc@0.5:  {metrics['acc@0.5']:.4f}")
        print(f"  Acc@0.7:  {metrics['acc@0.7']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Total refs evaluated: {metrics['num_samples']}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
