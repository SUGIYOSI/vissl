#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/finetuning/o.finetuning_simclr_deit_t16_in1k_50_to_cifar100__v1

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/fulltune/finetuning_simclr_deit.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/finetuning/finetuning_simclr_deit_t16_in1k_50_to_cifar100__v1" \
    config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.DATASET_NAMES=[CIFAR100] \
    config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar100/" \
    config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar100/" \
    config.DATA.TEST.DATASET_NAMES=[CIFAR100] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_in1k_50__v1/model_final_checkpoint_phase299.torch" \

echo '--End--'
echo `date`