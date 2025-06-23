#from src.api.llama4 import ask_llama
import json
import os

import requests
import time

REQUESTS_LIMIT = 29
TIME_WINDOW = 60  # secondi
requests_made = 0
start_time = time.time()




def ask_llama4(prompt):
    global requests_made
    
    # Se abbiamo raggiunto il limite di richieste...
    if requests_made >= REQUESTS_LIMIT:
        
        print(f"Limite richieste raggiunto. Attendo {TIME_WINDOW:.1f} secondi...")
        time.sleep(TIME_WINDOW)
        # Resetto il contatore e il timer
        requests_made = 0
        

    API_KEY = os.environ.get("LLAMA4_KEY", "")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)

    requests_made += 1

    if response.status_code == 200:
        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"Errore: risposta API priva di 'choices': {res_json}"
    else:
        return f"Errore HTTP {response.status_code}: {response.text}"


def process_ocr_file(input_file, gold_file, output_file, print_result=False):
    import json

    with open(input_file, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    results = []

    print("\n")
    print("|========================================")
    print("| \033[34mTranslating with llama4 ...\033[0m")

    for key in ocr_data:
        ocr_text = ocr_data.get(key, "")
        gold_text = gold_data.get(key, "")

        if not ocr_text:
            continue

        judge_prompt = "Correct the following text, fixing spelling and punctuation. Return only the corrected text, with no explanations or introductory phrases:"
        prompt = f"{judge_prompt}\n{ocr_text}"

        correction = ask_llama4(prompt)

        # Rimozione opzionale di frasi introduttive
        if isinstance(correction, str):
            correction = correction.replace("Ecco il testo corretto: ", "").strip()

        if print_result:
            print(f"ID {key}")
            print(f"Originale: {ocr_text}")
            print(f"Oro: {gold_text}")
            print(f"Correzione: {correction}\n")

        results.append({
            "ocr": ocr_text,
            "gold": gold_text,
            "correction": correction,
            "prometheus_score": None,
            "gemini_score": None,
            "human_score": None
        })

        time.sleep(1)

    # Scrivi come lista di dizionari
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print("|========================================\n")

    



## DA CONTROLLARE CHE LE CORREZIONI TALVOLTA INIZANO CON "ECCO IL TESTO CORRETTO:". Essendo sempre lo stesso pattern, si può facilmente rimuovere
        # Se vuoi rimuovere questa parte, puoi farlo con una regex o semplicemente con una slice
        # correction = correction.replace("Ecco il testo corretto: ", "").strip()

def correct_with_llama4(file_name, print_result=False):
    file = file_name
    datapath = "datasets/eng/"

    input_path = datapath + file + "_ocr.json"
    gold_path = datapath + file + "_clean.json"
    output_path = datapath + "corrections/llama4/" + file + ".json"

    process_ocr_file(input_path, gold_path, output_path, print_result)





