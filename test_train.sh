#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -o output/debug/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`

python run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/simclr/quick_1gpu_resnet50_simclr.yaml \
    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.CHECKPOINT.DIR="./checkpoints/debug_5" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

echo '--End--'
echo `date`