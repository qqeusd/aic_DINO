"""
Convert RefCOCO / RefCOCO+ / RefCOCOg to ODVG JSONL format.

RefCOCO data expected structure:
  {root}/
    refs(unc).p          # pickle or json: referring expressions (RefCOCO)
    refs(umd).p          # RefCOCOg (UMD split)
    refs(google).p       # RefCOCOg (Google split)
    instances.json       # COCO bbox annotations
    {image_dir}/         # COCO images (train2014)

Usage:
  python tools/refcoco2odvg.py \
    --ref_path data/refcoco/refs\(unc\).p \
    --instances_path data/refcoco/instances.json \
    --image_root data/refcoco/train2014 \
    --output_file data/refcoco/refcoco_train_odvg.jsonl \
    --split train
"""

import argparse
import json
import os
import pickle
from collections import defaultdict


def load_refs(ref_path):
    """Load referring expressions from .p (pickle) or .json file."""
    if ref_path.endswith('.p'):
        with open(ref_path, 'rb') as f:
            return pickle.load(f)
    elif ref_path.endswith('.json'):
        with open(ref_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ref_path}")


def load_instances(instances_path):
    """Load COCO instances (bbox annotations)."""
    with open(instances_path, 'r') as f:
        coco = json.load(f)
    # Build: ann_id -> annotation (with bbox, image_id, category_id)
    ann_by_id = {ann['id']: ann for ann in coco['annotations']}
    # Build: image_id -> image info
    img_by_id = {img['id']: img for img in coco['images']}
    # Build: cat_id -> cat_name
    cat_by_id = {cat['id']: cat['name'] for cat in coco['categories']}
    return ann_by_id, img_by_id, cat_by_id


def convert_refcoco(refs, ann_by_id, img_by_id, split_filter=None):
    """
    Group referring expressions by image_id, output ODVG format.

    Args:
        refs: list of ref objects with keys: ref_id, image_id, ann_id, split, sentences
        ann_by_id: dict ann_id -> COCO annotation
        img_by_id: dict image_id -> COCO image info
        split_filter: 'train', 'val', 'testA', 'testB' or None (all)

    Returns:
        List of ODVG dicts (one per image)
    """
    # Group refs by image_id
    image_refs = defaultdict(list)
    for ref in refs:
        if split_filter is not None and ref.get('split', '') != split_filter:
            continue
        image_id = ref['image_id']
        image_refs[image_id].append(ref)

    odvg_list = []
    skipped = 0

    for image_id, ref_list in image_refs.items():
        if image_id not in img_by_id:
            skipped += 1
            continue

        img_info = img_by_id[image_id]
        regions = []

        for ref in ref_list:
            ann_id = ref.get('ann_id')
            if ann_id is None or ann_id not in ann_by_id:
                skipped += 1
                continue

            ann = ann_by_id[ann_id]
            bbox_xywh = ann['bbox']  # COCO format: [x, y, w, h]
            x, y, w, h = bbox_xywh
            bbox_xyxy = [x, y, x + w, y + h]  # Convert to [x1, y1, x2, y2]

            # Get the first sentence for each ref (there could be multiple)
            for sent_info in ref.get('sentences', []):
                phrase = sent_info.get('sent', sent_info.get('raw', '')).strip()
                if phrase:
                    regions.append({
                        "bbox": bbox_xyxy,
                        "phrase": phrase,
                    })
                else:
                    skipped += 1

        if len(regions) == 0:
            skipped += 1
            continue

        # Build ODVG entry
        file_name = img_info.get('file_name', img_info.get('filename', f'COCO_train2014_{image_id:012d}.jpg'))
        odvg_entry = {
            "filename": file_name,
            "height": img_info['height'],
            "width": img_info['width'],
            "grounding": {
                "caption": " . ".join([r['phrase'] for r in regions]) + " .",
                "regions": regions,
            }
        }
        odvg_list.append(odvg_entry)

    print(f"  == Images: {len(odvg_list)}, Skipped: {skipped}")
    return odvg_list


def main():
    parser = argparse.ArgumentParser(description="Convert RefCOCO to ODVG JSONL format")
    parser.add_argument("--ref_path", type=str, required=True,
                        help="Path to refs(.p or .json) for RefCOCO/+/g")
    parser.add_argument("--instances_path", type=str, required=True,
                        help="Path to COCO instances.json (bbox annotations)")
    parser.add_argument("--image_root", type=str, default="",
                        help="Root directory of images (for reference only, not embedded in output)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--split", type=str, default=None,
                        help="Filter by split: train, val, testA, testB")
    parser.add_argument("--image_prefix", type=str, default="",
                        help="Optional prefix to prepend to filename (e.g. 'train2014/')")
    args = parser.parse_args()

    print(f"Loading refs from: {args.ref_path}")
    refs = load_refs(args.ref_path)
    print(f"  == Total refs loaded: {len(refs)}")

    print(f"Loading instances from: {args.instances_path}")
    ann_by_id, img_by_id, cat_by_id = load_instances(args.instances_path)
    print(f"  == Total images: {len(img_by_id)}, annotations: {len(ann_by_id)}")

    print(f"Converting (split={args.split or 'all'})...")
    odvg_data = convert_refcoco(refs, ann_by_id, img_by_id, split_filter=args.split)

    # Optionally prepend image_prefix to filenames
    if args.image_prefix:
        prefix = args.image_prefix.rstrip('/') + '/'
        for entry in odvg_data:
            entry['filename'] = prefix + entry['filename']

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        for entry in odvg_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(odvg_data)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
