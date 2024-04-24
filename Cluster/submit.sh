#!/bin/bash

#SBATCH -p gpu                   # Specify the GPU partition
#SBATCH --gres="gpu:k80:1"       # Request 1 Nvidia v100s GPU
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # Run 1 task
#SBATCH -c 1                     # Request 1 CPU core
#SBATCH --mem=64G                # Request 128GB of memory, as specified
#SBATCH -t 11:00:00              # Set maximum runtime to 1 hour
#SBATCH -J PyDenseJob            # Set the job name to "PyDenseJob"
#SBATCH -o PyDenseJob-%j.out     # Set the output file name, %j will be replaced with the job ID

# Load the necessary modules. Make sure these modules are available on your cluster.
module load Python/3.10         # Load Python 3.9, adjust according to your cluster's available modules
module load CUDA/11.7.0           # Load CUDA, if required by TensorFlow or other libraries
module load TensorFlow/
# Assuming you might want to install Python packages to a local directory
export PIP_TARGET=/home/$USER/.local

# Run your Python script
python3 Cluster.py

