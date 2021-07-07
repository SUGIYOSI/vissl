#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=05:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/VISSL/VISSL_386/bin/activate
module load cuda/10.2.89

echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
    hydra.verbose=true \
    config=/benchmark/fulltune/finetuning_simclr_deit.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug_finetuning_simclr_resnet_CIFAR10_v5" \
    config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TRAIN.DATASET_NAMES=[CIFAR10] \
    config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
    config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar10/" \
    config.DATA.TEST.DATASET_NAMES=[CIFAR10] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.OPTIMIZER.num_epochs=100 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_224_v1/model_phase31.torch" \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""

# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch"
# /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_deit_t16_224_v1/model_phase31.torch
echo '--End--'
echo `date`
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

# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch" \
# config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
# config.OPTIMIZER.num_epochs=3 \
# --------------------------------------------------------------------------------------------------------
# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=pretrain/finetuning_simclr_resnet50_in1k.yaml \
#     config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.DATASET_NAMES=[fakeimagenet_v1_1k] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug/debug_finetuning_simclr_resnet_CIFAR10_v1" \
#     config.OPTIMIZER.num_epochs=3 \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \

# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=/benchmark/fulltune/imagenet1k/finetuning_simclr_resnet50_in1k.yaml \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug/dg_finetuning_simclr_resnet_in1k_to_cifar100_v3" \
#     config.DATA.TRAIN.DATA_SOURCES=[torchvision_dataset] \
#     config.DATA.TRAIN.LABEL_SOURCES=[torchvision_dataset] \
#     config.DATA.TRAIN.DATASET_NAMES=[CIFAR100] \
#     config.DATA.TRAIN.COPY_DESTINATION_DIR="/tmp/cifar10/" \
#     config.DATA.TEST.DATA_SOURCES=[torchvision_dataset] \
#     config.DATA.TEST.LABEL_SOURCES=[torchvision_dataset] \
#     config.DATA.TEST.COPY_DESTINATION_DIR="/tmp/cifar10/" \
#     config.DATA.TEST.DATASET_NAMES=[CIFAR100] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/pretrain/pretrain_simclr_imagenet1k_v1/model_phase15.torch" \
#     config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \

# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=pretrain/supervised/supervised_4gpu_resnet_example \
#     config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TRAIN.DATA_PATHS=[/gs/hs0/tga-i/sugiyama.y.al/VISSL/other/dummy_data/train] \
#     config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2 \
#     config.DATA.TEST.DATA_SOURCES=[disk_folder] \
#     config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TEST.DATA_PATHS=[/gs/hs0/tga-i/sugiyama.y.al/VISSL/other/dummy_data/val] \
#     config.DATA.TEST.BATCHSIZE_PER_REPLICA=2 \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.OPTIMIZER.num_epochs=5 \
#     config.OPTIMIZER.param_schedulers.lr.values=[0.01,0.001] \
#     config.OPTIMIZER.param_schedulers.lr.milestones=[1] \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/test_2"

# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=pretrain/simclr/quick_1gpu_resnet50_simclr.yaml \
#     config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/debug_simclr_deit_imagenet1k_v6" \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=/test/integration_test/eval_in1k_linear_imagefolder_head.yaml \
#     config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2 \
#     config.DATA.TEST.DATA_SOURCES=[disk_folder] \
#     config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TEST.BATCHSIZE_PER_REPLICA=2 \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/test_3" \
#     config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/resnet50-19c8e357.pth" \
#     config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
#     config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

# python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/run_distributed_engines.py \
#     hydra.verbose=true \
#     config=/benchmark/fulltune/imagenet1k/eval_resnet_4gpu_transfer_in1k_fulltune.yaml \
#     config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2 \
#     config.DATA.TEST.DATA_SOURCES=[disk_folder] \
#     config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
#     config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
#     config.DATA.TEST.BATCHSIZE_PER_REPLICA=2 \
#     config.OPTIMIZER.num_epochs=2 \
#     config.OPTIMIZER.param_schedulers.lr.values=[0.01,0.001] \
#     config.OPTIMIZER.param_schedulers.lr.milestones=[1] \
#     config.DISTRIBUTED.NUM_NODES=1 \
#     config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
#     config.CHECKPOINT.DIR="./train_result/test_2" \
#     config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result/resnet50-19c8e357.pth" \
#     config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
#     config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME="" \
#     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true