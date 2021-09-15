#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=02:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v5/pretrain_simclr_simclr_deit_tiny_in1k_V100_node_4__v5.o

source /etc/profile.d/modules.sh
module load python/3.6/3.6.12
module load cuda/10.2/10.2.89
module load cudnn/7.6/7.6.5
module load nccl/2.7/2.7.8-1
module load openmpi/2.1.6
module load gcc/7.4.0

export PATH="/home/acd13476wd/anaconda3/bin:${PATH}"
source activate vissl_371

cd /groups/gcd50666/acd13476wd/VISSL/vissl

cat ${SGE_JOB_HOSTLIST} > ./hostfile
HOST=${HOSTNAME:0:5}
mpiexec --hostfile ./hostfile -np ${NHOSTS} python -B tools/run_distributed_engines.py config=pretrain/vision_transformer/simclr/simclr_deit_ti16.yaml config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR=/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v5/pretrain_simclr_simclr_deit_tiny_in1k_V100_node_4__v5 config.CHECKPOINT.DIR=/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v5/pretrain_simclr_simclr_deit_tiny_in1k_V100_node_4__v5 config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] config.DISTRIBUTED.NUM_NODES=4 config.DATA.NUM_DATALOADER_WORKERS=10 config.DISTRIBUTED.NUM_PROC_PER_NODE=4 config.DISTRIBUTED.HOST=${HOST} 