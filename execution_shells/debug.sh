#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:20:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v14/o.check_pretrain_v100_node_2__v003.o

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
# module load cuda/10.2.89 openmpi/3.1.4-opa10.10
module load cuda/10.2.89 openmpi/3.1.4-opa10.10 nccl/2.8.4
module list

export NUM_PROC=1
export NGPUS=2
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export PORT=":8888"
export NCCL_IB_DISABLE=1

echo '--Start--'
echo `date`
mpirun -npernode $NUM_PROC -np $NGPUS \
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/simclr/simclr_resnet50.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/other/v14/check_pretrain_v100_node_2__v003" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=4 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=5 \
    config.DISTRIBUTED.RUN_ID=$MASTER_ADDR$PORT \

echo '--End--'
echo `date`
    # config.DISTRIBUTED.RUN_ID=$MASTER_ADDR$PORT \
    # config.DISTRIBUTED.INIT_METHOD="file" \