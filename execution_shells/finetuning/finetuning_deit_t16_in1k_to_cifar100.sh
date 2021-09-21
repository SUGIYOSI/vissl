#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/finetuning/deit_t16/o.finetuning_simclr_deit_t16_in1k_per10_to_cifar100__v1

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
    config=/benchmark/fulltune/cifar100/finetuning_deit_t16_to_cifar100.yaml \
    config.CHECKPOINT.DIR="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/finetuning/deit_t16/finetuning_simclr_deit_t16_in1k_per10_to_cifar100__v1" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/train_result__tsubame/pretrain/simclr/pretrain_simclr_deit_t16_in1k_per10__v1/model_final_checkpoint_phase299.torch" \

echo '--End--'
echo `date`