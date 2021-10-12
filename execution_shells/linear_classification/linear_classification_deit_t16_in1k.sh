#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -v USE_BEEOND=1
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/linear_classification/deit_t16/o.linear_classification_simclr_deit_t16_in1k_per10_to_in1k__v1

# ======== env ========
. /etc/profile.d/modules.sh
module load cuda/10.2.89 cudnn/7.6 openmpi/3.1.4-opa10.10 gcc/8.3.0
export PATH="/gs/hs0/tga-i/sugiyama.y.al/anaconda3/bin:${PATH}"
source activate VISSL_py371_cu102_pyt171
module list

# ======== use beeond ========
echo '--Com Start--'
echo `date`
tar -xf /gs/hs0/tga-i/sugiyama.y.al/datasets/ILSVRC2012/originalimages.tar.gz --use-compress-program=pigz -C /beeond
echo '--Com End--'
echo `date`

# ======== script ========
echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/vissl/run_distributed_engines.py \
    config=/benchmark/linear_image_classification/imagenet1k/linear_classification_deit_t16_to_in1k.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/linear_classification/deit_t16/linear_classification_simclr_deit_t16_in1k_per10_to_in1k__v1" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/simclr/pretrain_simclr_deit_t16_in1k_per10__v1/model_final_checkpoint_phase299.torch" \
    config.DATA.TRAIN.DATASET_NAMES=[beeond_original_imagenet_1k] \
    config.DATA.TEST.DATASET_NAMES=[beeond_original_imagenet_1k] \

echo '--End--'
echo `date`