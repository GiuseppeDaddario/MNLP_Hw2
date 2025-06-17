#!/bin/bash
#SBATCH --job-name=t5-eval
#SBATCH --output=logs/t5_ocr.out
#SBATCH --error=logs/t5_ocr.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2               # 1 GPU basta per inferenza
#SBATCH --time=01:00:00
#SBATCH --partition=accelerated     # Cambia se la partizione si chiama diversamente
#SBATCH --mem=32G

# === Carica i moduli necessari ===
module load cuda/12.1
module load python/3.10

# === Attiva il tuo ambiente virtuale o Conda ===
source ~/envs/t5ocr/bin/activate  # Sostituisci col path corretto

# === Vai nella directory dove si trova lo script ===
cd $SLURM_SUBMIT_DIR

# === Avvia lo script di inferenza ===
python t5_evaluate.py
