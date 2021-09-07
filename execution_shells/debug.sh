#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=02:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v2/pretrain_simclr_resnet50_in1k_V100_node_1__v1.o

source /etc/profile.d/modules.sh
# MARK: for V100
# source /groups/gcd50666/acd13476wd/VISSL/VISSL_386/bin/activate
# module load openmpi/3.1.6 cuda/10.2/10.2.89
# MARK: for A100
source /groups/gcd50666/acd13476wd/VISSL/VISSL_A100_386/bin/activate
module load openmpi/3.1.6 cuda/11.1/11.1.1
module list

export NUM_PROC=1
export NGPUS=2

echo '--Start--'
echo `date`
mpirun -npernode $NUM_PROC -np $NGPUS \
python /groups/gcd50666/acd13476wd/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/simclr/simclr_resnet50.yaml \
    config.CHECKPOINT.DIR="/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v2/pretrain_simclr_resnet50_in1k_V100_node_1__v1" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=4 \

echo '--End--'
echo `date`