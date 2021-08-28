#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=00:30:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2999
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v1/check_pretrain_gpu_1__v1.o

source /groups/gcd50666/acd13476wd/VISSL/VISSL_386/bin/activate
module load cuda/10.2/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \
    config.CHECKPOINT.DIR="/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v1" \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k_10] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=99999 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.OPTIMIZER.num_epochs=30 \

echo '--End--'
echo `date`