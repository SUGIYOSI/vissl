#!/bin/bash
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=00:01:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -j y
#$ -o aaaa.o

echo '--Start--'
echo `date`

export MASTER_ADDR=`echo $(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1) | sed -e "s/[\r\n]\+//g"`
export RUN_ID=${MASTER_ADDR%% *}:8888
# `echo ${ADDR} | sed -e "s/[\r\n]\+//g"`

# export TEMP=${ADDR%%::*}
# export MASTER_ADDR=${TEMP%% *}
echo "[${MASTER_ADDR}]"
echo "[${RUN_ID}]"
# echo $MASTER_ADDR

# export GGG=${ADDR%\n*}
# echo "[${GGG}]"

# VAR=`echo ${ADDR} | sed -e "s/[\r\n]\+//g"`
# echo "[${VAR}]"


# export AAA="10.0.56.5 fe80::74b:bbb0:f321:e16f    "
# export BBB=${AAA%%::*}
# export CCC=${BBB%% *}
# echo $CCC
# # echo ${${AAA%%::*}%% *}
# echo ${"10.0.56.5 fe80::74b:bbb0:f321:e16f    "%%::*}


echo '--End--'
echo `date`