#!/bin/bash
# Training script for RefCOCO / RefCOCO+ / RefCOCOg visual grounding baseline
#
# Usage:
#   bash tools/train_refcoco.sh refcoco       # Train on RefCOCO, detect free GPUs automatically
#   bash tools/train_refcoco.sh refcoco 8     # Specify 8 GPUs (all)
#   bash tools/train_refcoco.sh refcoco 7 0,2,3,4,5,6,7  # 7 GPUs, exclude GPU 1
#   bash tools/train_refcoco.sh refcocog 4    # 4 GPUs

DATASET=${1:-refcoco}
GPU_NUM=${2:-8}
GPU_IDS=${3:-}

# Detect free GPUs if not specified
if [ -z "$GPU_IDS" ]; then
  # Find GPUs with <2GB used, take first GPU_NUM
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | \
    awk -F',' '{gsub(/ MiB/,"",$2); if($2<2000) print $1}' | \
    head -n ${GPU_NUM} | tr '\n' ',' | sed 's/,$//')

  if [ "$(echo $FREE_GPUS | tr ',' '\n' | wc -l)" -lt "$GPU_NUM" ]; then
    echo "ERROR: Need $GPU_NUM free GPUs, only $(echo $FREE_GPUS | tr ',' '\n' | wc -l) available"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
  fi
  GPU_IDS=$FREE_GPUS
fi

# Paths
CONFIG="config/cfg_refcoco_swinb.py"
DATASETS="config/datasets_${DATASET}.json"
OUTPUT_DIR="./output/${DATASET}_swinb_baseline"
PRETRAIN_MODEL="/home/zcoop8/zhangxianping/groundingdino/pretrained/groundingdino_swinb_cogcoor.pth"

# Clean old checkpoints and logs for fresh start
rm -f ${OUTPUT_DIR}/checkpoint*.pth
rm -f logs/events.out.*
mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "Training RefCOCO Baseline (Swin-B)"
echo "Dataset:   ${DATASET}"
echo "Config:    ${CONFIG}"
echo "GPUs:      ${GPU_IDS} (${GPU_NUM} total)"
echo "Output:    ${OUTPUT_DIR}"
echo "Pretrain:  ${PRETRAIN_MODEL}"
echo "AMP:       true"
echo "Batch:     2/GPU (effective batch = ${GPU_NUM}×2)"
echo "================================================"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_port=$((RANDOM + 10000)) \
    main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CONFIG} \
    --datasets ${DATASETS} \
    --pretrain_model_path "${PRETRAIN_MODEL}" \
    --amp \
    --num_workers 2

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"
