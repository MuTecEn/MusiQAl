#!/bin/bash

#SBATCH --job-name=trial0
#SBATCH --account=ec29
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=ifi_accel
# #SBATCH --gres=gpu:1
##SBATCH --qos=devel
# #SBATCH --mem=32G      
# #SBATCH --gpus=a100:1
#SBATCH --gpus=rtx30:1

module purge
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source /your/env/dir/avqaoriginal/bin/activate

DIR=/your/dir/to/AVST
