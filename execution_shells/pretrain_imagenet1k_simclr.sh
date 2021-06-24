#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=03:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`

python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/vision_transformer/simclr/simclr_deit_ti16.yaml \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet_1k] \
    config.DATA.TEST.DATA_SOURCES=[disk_folder] \
    config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
    config.DATA.TEST.DATASET_NAMES=[imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.CHECKPOINT.DIR="./train_result/pretrain_simclr_imagenet1k_v3" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

echo '--End--'
echo `date`
    # config=pretrain/vision_transformer/simclr/vit_t16.yaml \
    # config=pretrain/vision_transformer/simclr/simclr_deit_ti16.yaml \
    # config=pretrain/simclr/simclr_1node_resnet50.yaml \