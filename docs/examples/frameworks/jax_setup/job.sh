#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=unkillable

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

# Creating the environment for the first time:
# Jax comes with precompiled binaries targetting a specific version of CUDA. In
# case you encounter an error like the following:
#
# The NVIDIA driver's CUDA version is 11.7 which is older than the ptxas CUDA
# version (11.8.89). Because the driver is older than the ptxas version, XLA
# is disabling parallel compilation, which may slow down compilation. You
# should update your NVIDIA driver or use the NVIDIA-provided CUDA forward
# compatibility packages.
#
# Try installing the specified version of CUDA in conda :
# https://anaconda.org/nvidia/cuda. E.g. "nvidia/label/cuda-11.8.0" if ptxas
# CUDA version is 11.8.XX
#
# conda create -y -n jax -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
# conda activate jax
# virtualenv ~/jax
# source ~/jax/bin/activate
# pip install pip install --upgrade "jax[cuda11_pip]" \
#    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
#    pillow optax rich torch torchvision flax tqdm

# Activate the environment:
conda activate jax
source ~/jax/bin/activate


# Fixes issues with MIG-ed GPUs
unset CUDA_VISIBLE_DEVICES

python main.py
