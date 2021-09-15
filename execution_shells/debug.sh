#!/bin/bash
#$ -cwd
#$ -l rt_AF=2
#$ -l h_rt=00:20:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v6/pretrain_simclr_simclr_deit_tiny_in1k_A100_node_2__v2.o

# ======== env ========
# source /etc/profile.d/modules.sh
# module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 nccl/2.7/2.7.8-1 openmpi/2.1.6 gcc/7.4.0
# export PATH="/home/acd13476wd/anaconda3/bin:${PATH}"
# source activate vissl_371
# module list

# ======== env ========
source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1 cudnn/8.1/8.1.1 nccl/2.8/2.8.4-1 openmpi/3.1.6 gcc/9.3.0
export PATH="/home/acd13476wd/anaconda3/bin:${PATH}"
source activate VISSL_371_cuda_111
module list

# ======== configs ========
export NUM_NODES=2
export NUM_PROC=8
export BATCHSIZE_FOR_GPU=256
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
cd /groups/gcd50666/acd13476wd/VISSL/vissl
cat ${SGE_JOB_HOSTLIST} > ./hostfile
HOST=${HOSTNAME:0:5}
export NCCL_IB_DISABLE=1

# ======== script ========
echo '--Start--'
echo `date`
mpiexec --hostfile ./hostfile -np ${NHOSTS} \
python -B /groups/gcd50666/acd13476wd/VISSL/vissl/tools/run_distributed_engines.py \
    config=pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \
    config.CHECKPOINT.DIR="/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v6/pretrain_simclr_simclr_deit_tiny_in1k_A100_node_2__v2" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.HOST=${HOST} \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \

echo '--End--'
echo `date`

    # config=pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \
    # config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    # config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    # config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    # config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    # config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \