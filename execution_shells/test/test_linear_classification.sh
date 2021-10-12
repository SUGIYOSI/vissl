#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=24:00:00
#$ -j y
#$ -v USE_BEEOND=1
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/debug/v4/o.test_linear_classification_resnet50_in1k_to_in1k__v1

# ======== env ========
. /etc/profile.d/modules.sh
module load cuda/10.2.89 cudnn/7.6 openmpi/3.1.4-opa10.10 gcc/8.3.0
export PATH="/gs/hs0/tga-i/sugiyama.y.al/anaconda3/bin:${PATH}"
source activate VISSL_py371_cu102_pyt171
module list

# ======== configs ========
export NPERNODE=1
export NUM_NODES=2
export NUM_PROC=4
export BATCHSIZE_FOR_GPU=32
export BATCHSIZE=$(($BATCHSIZE_FOR_GPU*$NUM_NODES*$NUM_PROC))
export MASTER_ADDR=$(ip addr show dev ib0 | grep '\<inet\>' | cut -d " " -f 6 | cut -d "/" -f 1)
export RUN_ID=$MASTER_ADDR:8888
export NCCL_IB_DISABLE=1
export BEEONDDIR="/beeond"

# ======== use beeond ========
echo '--Com Start--'
echo `date`
tar -xf /gs/hs0/tga-i/sugiyama.y.al/datasets/ILSVRC2012/originalimages.tar.gz --use-compress-program=pigz -C $BEEONDDIR
echo '--Com End--'
echo `date`

# ======== script ========
echo '--Start--'
echo `date`
mpirun -npernode $NPERNODE -np $NUM_NODES \
python -B /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/linear_image_classification/imagenet1k/linear_classification_resnet50_to_in1k.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/debug/v4/test_linear_classification_resnet50_in1k_to_in1k__v1" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/simclr/pretrain_simclr_resnet50_in1k_per10__v1/model_final_checkpoint_phase199.torch" \
    config.DATA.TRAIN.DATASET_NAMES=[beeond_original_imagenet_1k] \
    config.DATA.TEST.DATASET_NAMES=[beeond_original_imagenet_1k] \
    config.DISTRIBUTED.NUM_NODES=$NUM_NODES \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=$NUM_PROC \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=$NUM_PROC \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=$BATCHSIZE_FOR_GPU \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=$BATCHSIZE \
    config.DISTRIBUTED.RUN_ID=$RUN_ID \

echo '--End--'
echo `date`