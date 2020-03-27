#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/dcase"

# You need to modify this path as your workspace
WORKSPACE="/home/dcase/pub_dcase_cnn"

DEV_SUBTASK_B_DIR="development-subtaskB-mobile"

BACKEND="pytorch"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE


############ Development subtask B ############
# Train model for subtask B
python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask B
python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=15000 --cuda



