import json
import os

#==== BUILDER FOR OUTPUT FILES ====
def build_output_file(FILE_NAME, correction_model):
    groupname = "C0rr3tt0r1_4ut0m4t1c1"
    JUDGE_PATH = f"outputs/{correction_model}/{groupname}-hw2_ocr-{correction_model}.json"
    CORRECTION_PATH = f"datasets/corrections/{correction_model}/{FILE_NAME}.json"

    # Load the JSON data
    with open(CORRECTION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Costruisci dizionario nel formato richiesto
    output_data = {
        str(idx): entry["correction"]
        for idx, entry in enumerate(data)
    }

    # Build the directory if it does not exist
    os.makedirs(os.path.dirname(JUDGE_PATH), exist_ok=True)

    #==== SAVING ====
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Output file saved in: {JUDGE_PATH}")
    #===============

#==============================================