#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/finetuning/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/fulltune/imagenet1k/finetuning_simclr_resnet50_in1k.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/finetuning/finetuning_simclr_resnet_in1k_to_cifar10_v1" \
    config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.DATASET_NAMES=[CIFAR100] \
    config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    config.DATA.TEST.DATASET_NAMES=[CIFAR100] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch" \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \

echo '--End--'
echo `date`