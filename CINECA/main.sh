#!/bin/bash
#SBATCH --job-name=minerva
#SBATCH --time=04:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:2                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --qos=normal                          # Quality of Service
#SBATCH --output=CINECA/logs/minerva_main.out
#SBATCH --error=CINECA/logs/minerva_main.err
#SBATCH --account=try25_navigli               # Project account number



# === Carica i moduli necessari ===
module load cuda/12.1
module load python/3.10

# === Attiva il tuo ambiente virtuale o Conda ===
source ~/mnlp/bin/activate  # Sostituisci col path corretto

# === Vai nella directory dove si trova lo script ===
cd $SLURM_SUBMIT_DIR

# === Avvia lo script di inferenza ===
python main.py
