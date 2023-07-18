#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
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


# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script
python main.py
