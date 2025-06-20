## Questo file serve per trasformare ICDAR dataset in un jsonL da dare in pasto a bart1

import os
import json

INPUT_DIR = "datasets/eng_monograph"  # Cambia con il path reale
OUTPUT_FILE = "icdar_eng_monograph.jsonl"

def clean_line(line: str) -> str:
    """Rimuove il tag iniziale (es. '[OCR_toInput]') e simboli di allineamento '@'"""
    return line.strip().split("] ", 1)[-1].replace("@", "").replace("#", "").strip()

def process_icdar_file(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None  # File malformato

    # Prendi solo il contenuto utile
    ocr_input = clean_line(lines[0])
    gold = clean_line(lines[2])
    
    return {"input": ocr_input, "target": gold}

def main():
    samples = []
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue
        full_path = os.path.join(INPUT_DIR, fname)
        result = process_icdar_file(full_path)
        if result:
            samples.append(result)

    print(f"Trovati {len(samples)} esempi validi. Salvataggio in {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    main()
