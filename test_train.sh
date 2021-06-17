#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`

# python run_distributed_engines.py \
#     hydra.verbose=true \
#     config=pretrain/simclr/quick_1gpu_resnet50_simclr.yaml \
#     config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.CHECKPOINT.DIR="./train_result/test_1" \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

python3 run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/supervised/supervised_4gpu_resnet_example \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
    config.DATA.TRAIN.DATA_PATHS=[/gs/hs0/tga-i/sugiyama.y.al/VISSL/other/dummy_data/train] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2 \
    config.DATA.TEST.DATA_SOURCES=[disk_folder] \
    config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
    config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
    config.DATA.TEST.DATA_PATHS=[/gs/hs0/tga-i/sugiyama.y.al/VISSL/other/dummy_data/val] \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=2 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.OPTIMIZER.num_epochs=5 \
    config.OPTIMIZER.param_schedulers.lr.values=[0.01,0.001] \
    config.OPTIMIZER.param_schedulers.lr.milestones=[1] \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.CHECKPOINT.DIR="./train_result/test_2"

echo '--End--'
echo `date`