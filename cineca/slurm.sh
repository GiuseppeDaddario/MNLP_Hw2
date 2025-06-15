#!/bin/bash


#SBATCH --job-name=multi_gpu_job              # Descriptive job name
#SBATCH --time=04:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --qos=normal                          # Quality of Service
#SBATCH --output=multiGPUJob.out              # File for standard output
#SBATCH --error=multiGPUJob.err               # File for standard error
#SBATCH --account=try25_navigli               # Project account number

# Load necessary modules (adjust to your environment)
module load python/3.11                       # Load Python module
module load cuda/12.2                         # Load CUDA toolkit
module load openmpi                           # Load MPI implementation

# Attiva l'ambiente virtuale
source ~/mnlp/bin/activate

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                        # Enable NCCL debugging (for multi-GPU communication)

# Launch the distributed GPU application
# Replace with your actual command (e.g., mpirun or srun)
cd /leonardo/home/userexternal/gdaddari/MNLP_Hw2
python main.py