#!/bin/bash
# Training script for RefCOCO / RefCOCO+ / RefCOCOg visual grounding baseline
#
# Usage:
#   bash tools/train_refcoco.sh refcoco       # Train on RefCOCO
#   bash tools/train_refcoco.sh refcocog      # Train on RefCOCOg
#   bash tools/train_refcoco.sh refcoco+      # Train on RefCOCO+

DATASET=${1:-refcoco}
GPU_NUM=${GPU_NUM:-4}
if [ $# -ge 2 ]; then GPU_NUM=$2; fi

# Configuration paths (modify these for your setup)
CONFIG="config/cfg_refcoco.py"
DATASETS="config/datasets_${DATASET}.json"
OUTPUT_DIR="./output/${DATASET}_baseline"
PRETRAIN_MODEL=""  # Set to pretrained checkpoint path if available

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "Training RefCOCO Baseline"
echo "Dataset: ${DATASET}"
echo "GPUs: ${GPU_NUM}"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

python -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_port=$((RANDOM + 10000)) \
    main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CONFIG} \
    --datasets ${DATASETS} \
    --pretrain_model_path "${PRETRAIN_MODEL}" \
    --options text_encoder_type="bert-base-uncased"

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"
