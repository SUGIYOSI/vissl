#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o /groups/gcd50666/acd13476wd/VISSL/vissl/execution_shells/output/debug/v3/pretrain_simclr_resnet50_in1k_V100_node_1__v10.o

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== env ========
source /etc/profile.d/modules.sh
# MARK: for V100
# source /groups/gcd50666/acd13476wd/VISSL/VISSL_386/bin/activate
# module load openmpi/3.1.6 cuda/10.2/10.2.89
# MARK: for A100
# source /groups/gcd50666/acd13476wd/VISSL/VISSL_A100_386/bin/activate
# module load cuda/11.1/11.1.1 openmpi/3.1.6
# MARK: New ENV
module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 nccl/2.7/2.7.8-1 openmpi/2.1.6 gcc/7.4.0
source /groups/gcd50666/acd13476wd/VISSL/ENVS/VISSL_371_cuda_102/bin/activate
module list
pip list


# ======== configs ========
export NPERNODE=1
export NUM_NODES=1
export NUM_PROC=4
export BATCHSIZE_FOR_GPU=32
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=`echo $(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1) | sed -e "s/[\r\n]\+//g"`
export RUN_ID=${MASTER_ADDR%% *}:8888

# ======== script ========
echo '--Start--'
echo `date`
mpirun -npernode $NPERNODE -np $NUM_NODES \
python /groups/gcd50666/acd13476wd/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/simclr/simclr_resnet50.yaml \
    config.CHECKPOINT.DIR="/groups/gcd50666/acd13476wd/VISSL/vissl/train_result/debug/other/v3/pretrain_simclr_resnet50_in1k_V100_node_1__v10" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
    config.DISTRIBUTED.RUN_ID=$RUN_ID \

echo '--End--'
echo `date`