#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 8
#SBATCH -A m1759
#SBATCH --time=0-02:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=alexjschuy@gmail.com
#SBATCH --mail-type=ALL

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load esslurm pytorch/v1.5.0-gpu cmake

srun python train.py train=slurm