#!/bin/bash
set -e

# Variabili
MODEL_NAME="sapienzanlp/Minerva-7B-instruct-v1.0"
CACHE_DIR="./src/models/minerva/cache"
WHEELHOUSE_DIR="./src/models/minerva/wheelhouse"

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