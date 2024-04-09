#!/bin/bash
HOME=/p/home/ritwik
EXP_NAME="swin_v2_tiny_1e-4_wu1-hyper_512_256_ssl0.75-64_alpha10"
PROJECT_DIR=/p/home/ritwik/dev/xLT/
PRETRAINED_CKPT_PATH=/p/path/to/pretrained_weights

CONSTRAINT=$1

### init virtual environment if needed  
source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate xt

cd $PROJECT_DIR

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
NUMEXPR_MAX_THREADS=128 \
WANDB_MODE=offline \
PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH \
EXP_NAME=$EXP_NAME \
PYTHONUNBUFFERED=1 \
python $PROJECT_DIR/launch_scripts/submitit_train_cluster.py \
    --job_dir /p/app/projects/nga-frontier/xlt-runs/jobs/$EXP_NAME \
    --constraint $CONSTRAINT \
    --qos frontier \
    --account ODEFN5169CYFZ \
    --nodes 1 \
    --config $PROJECT_DIR/config/swin-t/swin_v2_tiny_1e-4_hyper_512_256_ssl.yaml
