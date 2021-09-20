#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=02:00:00
#$ -j y
#$ -v USE_BEEOND=1
#$ -o /gs/hs0/tga-i/sugiyama.y.al/VISSL/vissl/output__tsubame/debug/v3/o.test_comp

export BEEONDDIR="/beeond"

# ======== use beeond ========
echo '--Com Start--'
echo `date`
tar -xf /gs/hs0/tga-i/sugiyama.y.al/datasets/ILSVRC2012/originalimages.tar.gz --use-compress-program=pigz -C $BEEONDDIR
echo '--Com End--'
echo `date`

### debug ###
ls $BEEONDDIR
ls $BEEONDDIR/originalimages
cd $BEEONDDIR/originalimages
ls -1UR | wc -l