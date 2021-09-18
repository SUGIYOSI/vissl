#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -j y
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/execution_shells/output/debug/v2/o.pretrain_simclr_deit_in1k_p100_node_2__exampele.o

# ======== env ========
source /etc/profile.d/modules.sh
module load cuda/11.0.194 gcc/8.3.0 cudnn/8.1 nccl/2.8.4 openmpi/3.1.4-opa10.10
export PATH="/home/acd13476wd/anaconda3/bin:${PATH}"
source activate VISSL_py371_cu110_pyt171
module list

cd /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl
cat ${SGE_JOB_HOSTLIST} > ./hostfile
HOST=${HOSTNAME:0:5}

echo "HOST: "
echo $HOST

echo "SGE_JOB_HOSTLIST: "
echo ${SGE_JOB_HOSTLIST}

echo "NHOSTS: "
echo ${NHOSTS}

echo '--End--'
echo `date`