#!/bin/bash
#$ -cwd
#$ -l f_node=4
#$ -l h_rt=02:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/debug/v2/o.pretrain_simclr_deit_in1k_p100_node_4__v2

# ======== env ========
. /etc/profile.d/modules.sh
module load cuda/11.0.194 gcc/8.3.0 cudnn/8.1 nccl/2.8.4 openmpi/3.1.4-opa10.10
export PATH="/gs/hs0/tga-i/sugiyama.y.al/anaconda3/bin:${PATH}"
source activate VISSL_py371_cu110_pyt171
module list

# ======== configs ========
export NPERNODE=1
export NUM_NODES=4
export NUM_PROC=4
export BATCHSIZE_FOR_GPU=128
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export RUN_ID=$MASTER_ADDR:8888
export NCCL_IB_DISABLE=1

# ======== script ========
echo '--Start--'
echo `date`
mpirun -npernode $NPERNODE -np $NUM_NODES \
python -B /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/debug/v2/pretrain_simclr_deit_in1k_p100_node_4__v2" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
    config.DISTRIBUTED.RUN_ID=$RUN_ID \

echo '--End--'
echo `date`

# ======== configs ========
# config=pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \