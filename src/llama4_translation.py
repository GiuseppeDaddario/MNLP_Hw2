#from src.api.llama4 import ask_llama
import json
import requests
import time
from src.post_process import textCleaner




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
        

    API_KEY = "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"
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


def process_ocr_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        ocr_text = item.get("ocr", "")
        if not ocr_text:
            continue

        # Costruisci il prompt per chiedere la correzione
        #judge_prompt =
        #judge_prompt =
        judge_prompt = "Correct the following text, fixing spelling and punctuation. Return only the corrected text, with no explanations or introductory phrases:"
        #judge_prompt = "Correggi il seguente testo rispettando ortografia e punteggiatura. Restituisci solo il testo corretto, senza alcuna spiegazione o introduzione:"

        prompt = f"{judge_prompt}\n{ocr_text}"
        
        correction = ask_llama4(prompt)
        print(f"Originale: {ocr_text}")
        print(f"Correzione: {correction}\n")

        item["llama4_correction"] = correction
        results.append(item)

        # Pausa breve per non sovraccaricare l'API (regola come ti serve)
        time.sleep(1)

    # Scrivi il file di output
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)




## DA CONTROLLARE CHE LE CORREZIONI TALVOLTA INIZANO CON "ECCO IL TESTO CORRETTO:". Essendo sempre lo stesso pattern, si può facilmente rimuovere
        # Se vuoi rimuovere questa parte, puoi farlo con una regex o semplicemente con una slice
        # correction = correction.replace("Ecco il testo corretto: ", "").strip()

def translate_with_llama4(file_name):
    
    file = file_name   # sostituisci solo il nome del file senza estensione
    
    datapath = "datasets/eng/"
    input_path = datapath + file +".json" # percorso del file di input
    output_path = datapath+"corrections/"+file+"_correction.json"  # nome del file output

    ## Call agli API di llama4 e creazione file
    process_ocr_file(input_path, output_path)

    ## Rimuove eventuale "Ecco il testo corretto: " dalle correzioni
    #textCleaner(output_path, output_path)





