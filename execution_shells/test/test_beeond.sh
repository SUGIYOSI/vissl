#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v1/o.pretrain_simclr_deit_t16_in1k_p100_node_1_beeyond__v1

# ======== env ========
source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386_cuda_102/bin/activate
module load cuda/10.2.89 cudnn/7.6 openmpi/3.1.4-opa10.10 gcc/8.3.0-cuda
module list

# ======== configs ========
export NPERNODE=1
export NUM_NODES=1
export NUM_PROC=4
export BATCHSIZE_FOR_GPU=64
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export RUN_ID=$MASTER_ADDR:8888

# ======== script ========
echo '--Start--'
echo `date`
mpirun -npernode $NPERNODE -np $NUM_NODES \
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/debug/simclr_deit_ti16.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/other/v1/pretrain_simclr_deit_t16_in1k_p100_node_1_beeyond__v1" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
    config.DISTRIBUTED.RUN_ID=$RUN_ID \

echo '--End--'
echo `date`