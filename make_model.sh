#!/bin/bash

#PBS -q h-small

#PBS -Wgroup_list=jh170036h

#PBS -l select=1:mpiprocs=36

#PBS -l walltime=47:50:00

cd $PBS_O_WORKDIR

. /etc/profile.d/modules.sh

module load anaconda3
module load intel/18.1.163 cuda/8.0.44-cuDNN6 keras/2.1.2





python make_model.py \
iteration_num=10 \
patience=64 \
tn_ratio=1 \
epoch_num_fix=24 \
scale_int=1 \
train_mode=all \
size_threshold=300 \
size_lowerlimit=1 \
threshold1=0.99 \
group_froc=validation \
width=15 \
> make_model_output \
