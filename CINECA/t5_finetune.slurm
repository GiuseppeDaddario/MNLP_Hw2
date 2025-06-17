#!/bin/bash
#SBATCH --job-name=t5-eval
#SBATCH --time=04:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:2                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --qos=normal                          # Quality of Service
#SBATCH --output=logs/t5_ocr.out
#SBATCH --error=logs/t5_ocr.err
#SBATCH --account=try25_navigli               # Project account number

# === Imposta ambiente ===
module load cuda/12.1  # o il modulo corretto per il tuo sistema
module load python/3.10

# === Attiva virtualenv/conda ===
source ~/envs/t5ocr/bin/activate   # cambia con il tuo path

# === Vai nella directory di lavoro ===
cd $SLURM_SUBMIT_DIR

# === Avvia il training ===
python t5_finetune.py
