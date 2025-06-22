import json
import os

def build_output_file(FILE_NAME, correction_model):
    groupname = "C0rr3tt0r1_4ut0m4t1c1"
    JUDGE_PATH = f"outputs/{correction_model}/{groupname}-hw2_ocr-{correction_model}.json"
    CORRECTION_PATH = f"datasets/eng/corrections/{correction_model}/{FILE_NAME}.json"

    # Carica dati dal file sorgente
    with open(CORRECTION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Costruisci dizionario nel formato richiesto
    output_data = {
        str(idx): entry["correction"]
        for idx, entry in enumerate(data)
    }

    # Crea la directory di output se non esiste
    os.makedirs(os.path.dirname(JUDGE_PATH), exist_ok=True)

    # Salva nel nuovo file
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Output file saved in: {JUDGE_PATH}")