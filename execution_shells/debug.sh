#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=03:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v12/o.debug_No0

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/pretrain/vision_transformer/simclr/simclr_deit_t16.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/other/v12/debug_No0" \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k_10] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_lr_batch_size=99999 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.OPTIMIZER.num_epochs=30 \

echo '--End--'
echo `date`
# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
# --------------------------------------------------------------------------------------------------------
# config=pretrain/vision_transformer/simclr/simclr_deit_ti16.yaml \
# config=pretrain/simclr/simclr_resnet50.yaml \
# config=/benchmark/fulltune/finetuning_simclr_resnet.yaml \
# config=/benchmark/fulltune/finetuning_simclr_deit.yaml \

# config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
# config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
# config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
# config.DATA.TRAIN.DATASET_NAMES=[CIFAR10] \
# config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar10/" \
# config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
# config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
# config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \

# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_in1k__v1/model_phase50.torch"
# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_in1k__v1/model_phase33.torch" \
# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch"
# /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_224_v1/model_phase31.torch
# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch" \
# config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
# config.OPTIMIZER.num_epochs=3 \
# --------------------------------------------------------------------------------------------------------
# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=/benchmark/fulltune/finetuning_simclr_deit.yaml \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug/v9/dg_finetuning_simclr_deit_t16_in1k_to_cifar100__v1" \
#     config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
#     config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
#     config.DATA.TRAIN.DATASET_NAMES=[CIFAR100] \
#     config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar100/" \
#     config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
#     config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
#     config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar100/" \
#     config.DATA.TEST.DATASET_NAMES=[CIFAR100] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.OPTIMIZER.num_epochs=100 \
#     config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_in1k__v1/model_phase50.torch"

    # config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.DATASET_NAMES=[CIFAR10] \
    # config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    # config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    # config.DATA.TEST.DATASET_NAMES=[CIFAR10] \
