#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/linear_classification/o.benchmark_linear_classification_resnet50_in1k_ep100_to_in1k_ep28__v1

# source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
source /gs/hs0/tga-i/sugiyama.y.al/VISSL/Debug_v1_VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/linear_classification/benchmark_linear_classification_resnet50_in1k_ep100_to_in1k_ep28__v1" \
    config.DATA.TRAIN.DATASET_NAMES=[original_imagenet_1k] \
    config.DATA.TEST.DATASET_NAMES=[original_imagenet_1k] \
    config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=4 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/distribution/simclr/model_final_checkpoint_phase99.torch" \

echo '--End--'
echo `date`

    # config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TRAIN.DATASET_NAMES=[CIFAR10] \
    # config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    # config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/im1k/" \
    # config.DATA.TEST.DATASET_NAMES=[CIFAR10] \