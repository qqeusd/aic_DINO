#!/bin/bash
# Training script for RefCOCO / RefCOCO+ / RefCOCOg visual grounding baseline
#
# Usage:
#   bash tools/train_refcoco.sh refcoco       # Train on RefCOCO (Swin-B)
#   bash tools/train_refcoco.sh refcocog      # Train on RefCOCOg (Swin-B)
#   bash tools/train_refcoco.sh refcoco+      # Train on RefCOCO+ (needs datasets_refcoco+.json)

DATASET=${1:-refcoco}
GPU_NUM=${GPU_NUM:-4}
if [ $# -ge 2 ]; then GPU_NUM=$2; fi

# Paths
CONFIG="config/cfg_refcoco_swinb.py"
DATASETS="config/datasets_${DATASET}.json"
OUTPUT_DIR="./output/${DATASET}_swinb_baseline"
PRETRAIN_MODEL="/home/zcoop8/zhangxianping/groundingdino/pretrained/groundingdino_swinb_cogcoor.pth"

mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "Training RefCOCO Baseline (Swin-B)"
echo "Dataset: ${DATASET}"
echo "Config:  ${CONFIG}"
echo "GPUs:    ${GPU_NUM}"
echo "Output:  ${OUTPUT_DIR}"
echo "Pretrain: ${PRETRAIN_MODEL}"
echo "================================================"

python -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_port=$((RANDOM + 10000)) \
    main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CONFIG} \
    --datasets ${DATASETS} \
    --pretrain_model_path "${PRETRAIN_MODEL}"

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"
