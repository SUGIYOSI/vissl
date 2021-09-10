#!/bin/bash
#$ -cwd
#$ -l rt_AF=2
#$ -l h_rt=05:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v2/pretrain_dino_deits16_in1k_A100_node_2__v1.o

# ======== env ========
source /etc/profile.d/modules.sh
# MARK: for V100
# source /groups/gcd50666/acd13476wd/VISSL/VISSL_386/bin/activate
# module load openmpi/3.1.6 cuda/10.2/10.2.89
# MARK: for A100
source /groups/gcd50666/acd13476wd/VISSL/VISSL_A100_386/bin/activate
module load cuda/11.1/11.1.1 openmpi/3.1.6
module list

# ======== configs ========
export NPERNODE=1
export NUM_NODES=2
export NUM_PROC=8
export BATCHSIZE_FOR_GPU=64
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=`echo $(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1) | sed -e "s/[\r\n]\+//g"`
export RUN_ID=${MASTER_ADDR%% *}:8888

# ======== script ========
echo '--Start--'
echo `date`
mpirun -npernode $NPERNODE -np $NUM_NODES \
python /groups/gcd50666/acd13476wd/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/dino/dino_16gpus_deits16.yaml \
    config.CHECKPOINT.DIR="/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v2/pretrain_dino_deits16_in1k_A100_node_2__v1" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
    config.DISTRIBUTED.RUN_ID=$RUN_ID \

echo '--End--'
echo `date`