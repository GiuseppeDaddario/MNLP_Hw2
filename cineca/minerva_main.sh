#!/bin/bash


#SBATCH --job-name=finetune_max
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --time=00:50:00
#SBATCH --output=./logs/max_job.out
#SBATCH --error=./logs/max_job.err
#SBATCH --account=try25_navigli

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
source ~/mnlp/bin/activate

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                        # Enable NCCL debugging (for multi-GPU communication)

# Launch the distributed GPU application
# Replace with your actual command (e.g., mpirun or srun)
cd /leonardo/home/userexternal/gdaddari/MNLP_Hw2
python src/models/minerva/main.py