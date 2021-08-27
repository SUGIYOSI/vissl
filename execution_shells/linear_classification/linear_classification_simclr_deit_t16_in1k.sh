#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/linear_classification/debug_v2/o.linear_classification_simclr_deit_t16_in1k_to_in1k_10__v1

# source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
source /gs/hs0/tga-i/sugiyama.y.al/VISSL/Debug_v1_VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/linear_image_classification/imagenet1k/linear_classification_simclr_deit_to_in1k.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug/linear_classification/v2/linear_classification_simclr_deit_t16_in1k_to_in1k_10__v1" \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k_10] \
    config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/in1k/" \
    config.DATA.TEST.DATA_SOURCES=[disk_folder] \
    config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
    config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/in1k/" \
    config.DATA.TEST.DATASET_NAMES=[original_imagenet_1k_10] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_in1k__v1/model_phase99.torch" \

echo '--End--'
echo `date`

    # config.OPTIMIZER.num_epochs=300 \
    # config.MODEL.AMP_PARAMS.USE_AMP=True \
    # config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch \

    # config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.DATASET_NAMES=[CIFAR10] \
    # config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATASET_NAMES=[CIFAR10] \

    # config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    # config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    # config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k_10] \
    # config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATA_SOURCES=[disk_folder] \
    # config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
    # config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATASET_NAMES=[original_imagenet_1k_10] \