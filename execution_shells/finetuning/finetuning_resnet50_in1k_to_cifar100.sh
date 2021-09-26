#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=16:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/finetuning/resnet50/o.finetuning_simclr_resnet50_in1k_per10_to_cifar100__v1

# ======== env ========
. /etc/profile.d/modules.sh
module load cuda/10.2.89 cudnn/7.6 openmpi/3.1.4-opa10.10 gcc/8.3.0
export PATH="/gs/hs0/tga-i/sugiyama.y.al/anaconda3/bin:${PATH}"
source activate VISSL_py371_cu102_pyt171
module list

# ======== script ========
echo '--Start--'
echo `date`
python /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/vissl/run_distributed_engines.py \
    config=/benchmark/fulltune/cifar100/finetuning_resnet50_to_cifar100.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/finetuning/resnet50/finetuning_simclr_resnet50_in1k_per10_to_cifar100__v1" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/simclr/pretrain_simclr_resnet50_in1k_per10__v1/model_final_checkpoint_phase199.torch" \

echo '--End--'
echo `date`

# ======== configs ========
# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/distribution/simclr/model_final_checkpoint_phase199.torch" \
# config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/simclr/pretrain_simclr_resnet50_in1k_per10__v1/model_final_checkpoint_phase199.torch" \