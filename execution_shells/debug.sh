#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -v USE_BEEOND=1
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v2/o.hahaha

# ======== Pyenv/ ========
# export PYENV_ROOT=$HOME/.pyenv
# export PATH=$PYENV_ROOT/bin:$PATH
# eval "$(pyenv init -)"

# ======== env ========
# source /gs/hs0/tga-i/sugiyama.y.al/VISSL/ENVS/VISSL_371_cuda102/bin/activate
# source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386_cuda_102/bin/activate
# module load cuda/10.2.89 cudnn/7.6 openmpi/3.1.4-opa10.10 gcc/8.3.0

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89 openmpi/3.1.4-opa10.10
module list

module load cuda/11.0.194 gcc/8.3.0 cudnn/8.1 nccl/2.8.4 openmpi/3.1.4-opa10.10

# ======== configs ========
export NPERNODE=1
export NUM_NODES=4
export NUM_PROC=4
export BATCHSIZE_FOR_GPU=32
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export RUN_ID=$MASTER_ADDR:8888


cat ${SGE_JOB_HOSTLIST} > ./hostfile
HOST=${HOSTNAME:0:5}
echo "HOST: "
echo $HOST

echo "HOME: "
echo $HOME
echo "SGE_BEEONDDIR: "
echo $SGE_BEEONDDIR

# ======== script ========
# echo '--Start--'
# echo `date`
# mpirun -npernode $NPERNODE -np $NUM_NODES \
# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=pretrain/simclr/simclr_resnet50.yaml \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug/other/v1/pretrain_simclr_resnet50_in1k_p100_node_4__v8" \
#     config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
#     config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
#     config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
#     config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
#     config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
#     config.DISTRIBUTED.RUN_ID=$RUN_ID \

# echo '--End--'
# echo `date`
    # config=/pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \