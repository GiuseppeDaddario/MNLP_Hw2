#!/bin/bash


#SBATCH --job-name=multi_gpu_job              # Descriptive job name
#SBATCH --time=04:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:2                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --qos=normal                          # Quality of Service
#SBATCH --output=multiGPUJob.out              # File for standard output
#SBATCH --error=multiGPUJob.err               # File for standard error
#SBATCH --account=try25_navigli               # Project account number

# Load necessary modules (adjust to your environment)
module unload libiconv || true
module unload libxml2 || true
module unload gettext || true

module load libiconv/1.17-nhc3mhm
module load libxml2/2.10.3-zbbe7lm
module load gettext/0.22.3-2g7elif
module load python/3.11.6--gcc--8.5.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0

# Attiva l'ambiente virtuale
source ~/prometheus-env/bin/activate

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                        # Enable NCCL debugging (for multi-GPU communication)

# Launch the distributed GPU application
# Replace with your actual command (e.g., mpirun or srun)
cd /leonardo/home/userexternal/lbenucci/MNLP_Hw2
python src/models/prometheus/evaluate.py