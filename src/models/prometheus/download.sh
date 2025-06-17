#!/bin/bash
set -e

# Variabili
MODEL_NAME="Unbabel/M-Prometheus-7B"
CACHE_DIR="./src/models/prometheus/cache"
WHEELHOUSE_DIR="./src/models/prometheus/wheelhouse"

# Crea cartelle se non esistono
mkdir -p "${CACHE_DIR}"
mkdir -p "${WHEELHOUSE_DIR}"

# Scarica modello e tokenizer con Transformers
echo "Scaricando modello e tokenizer nella cache locale..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', cache_dir='${CACHE_DIR}', trust_remote_code=True)
AutoTokenizer.from_pretrained('${MODEL_NAME}', cache_dir='${CACHE_DIR}', trust_remote_code=True)
"