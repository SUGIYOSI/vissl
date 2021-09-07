#!/bin/bash
#$ -cwd
#$ -l f_node=4
#$ -l h_rt=15:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v15/o.pretrain_simclr_deit_t16_in1k_p100_node_4__v1

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89 openmpi/3.1.4-opa10.10
module list

export NUM_PROC=1
export NGPUS=4
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export PORT=":8888"
export NCCL_IB_DISABLE=1

echo '--Start--'
echo `date`
mpirun -npernode $NUM_PROC -np $NGPUS \
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/simclr/simclr_resnet50.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/other/v15/pretrain_simclr_deit_t16_in1k_p100_node_4__v1" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=4 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=4 \
    config.DISTRIBUTED.RUN_ID=$MASTER_ADDR$PORT \

echo '--End--'
echo `date`
    # config.DISTRIBUTED.RUN_ID=$MASTER_ADDR$PORT \
    # config.DISTRIBUTED.INIT_METHOD="file" \