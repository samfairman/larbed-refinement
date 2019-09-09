#!/bin/bash

#SBATCH --exclude=sem-icefield,sem-iceglacier
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/include/c++/5

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64

cd /testpool/ops/samfairman/larbed-refinement

## run the actual reconstruction 
make

cd /testpool/ops/samfairman/larbed-refinement/bin
time ./direct_cuda_pattern
