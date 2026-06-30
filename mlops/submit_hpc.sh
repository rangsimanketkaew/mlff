#!/bin/bash

#SBATCH --job-name=mlff-training     # Job name
#SBATCH --nodes=2                    # Run on 2 nodes
#SBATCH --ntasks-per-node=4          # 4 tasks per node (1 per GPU)
#SBATCH --gres=gpu:4                 # Request 4 GPUs per node
#SBATCH --cpus-per-task=8            # CPU cores per GPU task
#SBATCH --mem=128G                   # Memory per node
#SBATCH --time=24:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=gpu              # GPU partition
#SBATCH --output=logs/train_%j.log   # Standard output and error log
#SBATCH --error=logs/train_%j.err    # Separate error log

# Exit immediately if any command fails
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment modules (HPC-specific, customize for your cluster)
module purge
module load gcc/10.2.0
module load cuda/11.8.0
module load openmpi/4.1.1
module load cudnn/8.8.0

# Activate Python virtual environment or Conda environment
source activate mlff-env

# --- PyTorch Distributed Environment Setup ---
# Find the primary node's IP address to act as Master Coordinator
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Resolve host to IP address
export MASTER_PORT=29500

# Total number of GPUs across all nodes
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))

echo "Distributed Multi-Node Run Details:"
echo "-----------------------------------"
echo "Master Address: $MASTER_ADDR"
echo "Master Port:    $MASTER_PORT"
echo "Nodes Allocated:$SLURM_JOB_NUM_NODES"
echo "GPUs per Node:  $SLURM_GPUS_ON_NODE"
echo "Total World Size: $WORLD_SIZE"
echo "-----------------------------------"

# Launch training via PyTorch torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    mlops/train_pipeline.py \
    --data_dir data \
    --model_path data/model_checkpoint.pt \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --energy_weight 1.0 \
    --force_weight 20.0 \
    --mlflow_uri http://mlflow.internal.hpc.org:5000
