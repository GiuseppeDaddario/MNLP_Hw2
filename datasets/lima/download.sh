#!/bin/bash
mkdir -p datasets/lima
echo "Downloading LIMA dataset from HuggingFace (GAIR/lima)..."
python3 -c "
from datasets import load_dataset
dataset = load_dataset('GAIR/lima')
dataset.save_to_disk('./datasets/lima')
"
echo "LIMA dataset downloaded and saved"