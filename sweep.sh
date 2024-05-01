#!/bin/sh

MODEL_SIZE="gpt2-medium"
DATASET_NAME="boolq"
BATCH_SIZE=32
LEARNING_RATE=1e-05
EPOCHS=3
SWEEP_SUBFOLDER="boolq_experiment_01"
LOSS_TYPE="xent"
MODEL_CKPT="gpt2-boolq"

# paths
RESULTS_FOLDER="./results"
DATASET_FOLDER="$RESULTS_FOLDER/$DATASET_NAME"
MODEL_FOLDER="$DATASET_FOLDER/bs=$BATCH_SIZE-dn=$DATASET_NAME-e=$EPOCHS-l=$LOSS_TYPE-lr=$LEARNING_RATE-ms=$MODEL_SIZE"
STRONG_CKPT_PATH="$MODEL_FOLDER/model.safetensors"
WEAK_LABELS_PATH="$MODEL_FOLDER/weak_labels"

# STRONG_CKPT_PATH="./results/$SWEEP_SUBFOLDER/bs=$BATCH_SIZE-dn=$DATASET_NAME-e=$EPOCHS-l=$LOSS_TYPE-l=$LEARNING_RATE"
# WEAK_LABELS_PATH="./results/$SWEEP_SUBFOLDER/bs=$BATCH_SIZE-dn=$DATASET_NAME-e=$EPOCHS-l=$LOSS_TYPE-l=$LEARNING_RATE-mc=$MODEL_CKPT/weak_labels"

python train_simple.py \
    --model_size=$MODEL_SIZE \
    --ds_name=$DATASET_NAME \
    --batch_size=$BATCH_SIZE \
    --lr=$LEARNING_RATE \
    --epochs=$EPOCHS \
    --sweep_subfolder=$SWEEP_SUBFOLDER \
    --loss=$LOSS_TYPE \
    --model_ckpt=$MODEL_CKPT \
    --strong_ckpt_path=$STRONG_CKPT_PATH \
    --weak_labels_path=$WEAK_LABELS_PATH
    