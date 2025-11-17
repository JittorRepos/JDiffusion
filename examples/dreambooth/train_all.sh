#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="Charles-Elena/stable-diffusion-2-1"
BASE_INSTANCE_DIR="the-path-to-dataset"
OUTPUT_DIR_PREFIX="style/style_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=500
SEED=0
GPU_COUNT=2
MAX_NUM=42

for ((folder_number = 0; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$gpu_id
        PROMPT=$(printf "style_%02d" $current_folder_number)

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt=$PROMPT \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --max_train_steps=$MAX_TRAIN_STEPS \
            --seed=$SEED"

        eval $COMMAND &
        sleep 3
    done
    wait
done
