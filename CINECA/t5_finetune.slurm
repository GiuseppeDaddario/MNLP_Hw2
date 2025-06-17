#!/bin/bash
#SBATCH --job-name=t5-ocr-correction
#SBATCH --output=logs/t5_ocr.out
#SBATCH --error=logs/t5_ocr.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2                # Usa 2 GPU A100
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated     # O specifica la coda usata da Cineca
#SBATCH --mem=128G

# === Imposta ambiente ===
module load cuda/12.1  # o il modulo corretto per il tuo sistema
module load python/3.10

# === Attiva virtualenv/conda ===
source ~/envs/t5ocr/bin/activate   # cambia con il tuo path

# === Vai nella directory di lavoro ===
cd $SLURM_SUBMIT_DIR

# === Avvia il training ===
python t5_finetune.py
