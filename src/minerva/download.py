import os
import subprocess
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sapienzanlp/Minerva-7B-instruct-v1.0"
CACHE_DIR = "./minerva-cache"
WHEELHOUSE_DIR = "./wheelhouse"
ARCHIVE_NAME = "minerva_offline_package.tar.gz"

def download_model():
    print(f"Scaricando modello e tokenizer {MODEL_NAME}...")
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

def download_wheels():
    print("Scaricando pacchetti Python necessari come wheel...")
    os.makedirs(WHEELHOUSE_DIR, exist_ok=True)
    packages = ["transformers", "peft", "datasets", "accelerate", "bitsandbytes"]
    cmd = ["pip", "download", *packages, "-d", WHEELHOUSE_DIR]
    subprocess.run(cmd, check=True)

def create_archive():
    print(f"Comprimo {CACHE_DIR} e {WHEELHOUSE_DIR} in {ARCHIVE_NAME}...")
    with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
        tar.add(CACHE_DIR)
        tar.add(WHEELHOUSE_DIR)
    print("Archivio creato con successo.")

if __name__ == "__main__":
    download_model()
    download_wheels()
    create_archive()
    print(f"Tutto pronto! Trasferisci '{ARCHIVE_NAME}' su Leonardo con scp o rsync.")