#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/pretrain/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/vision_transformer/simclr/deit_t16.yaml \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_224_v1" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

echo '--End--'
echo `date`