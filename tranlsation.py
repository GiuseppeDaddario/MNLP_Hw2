#from src.api.llama4 import ask_llama
import json
import requests
import time
import requests


import time
import requests

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
        prompt = f"Correggi rispettando la punteggiatura e l'ortografia, senza scrivere altro:\n{ocr_text}"
        
        correction = ask_llama4(prompt)
        print(f"Originale: {ocr_text}")
        print(f"Correzione: {correction}\n")

        results.append({
            "ocr": ocr_text,
            "llama4_correction": correction
        })

        # Pausa breve per non sovraccaricare l'API (regola come ti serve)
        time.sleep(1)

    # Scrivi il file di output
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    input_path = "datasets/eng/human_data.json"   # sostituisci con il nome del file input
    output_path = "datasets/eng/corrections/human_data_correction.json"  # nome del file output

    process_ocr_file(input_path, output_path)


#judge_prompt = "Correggi rispettando la punteggiatura, senza scrivere altro:" ## Insert here your prompt
#message_prompt = "The universa1 belief js, that a person sucked by a vampyre becomes a vampyre himself, arid sucks in his turn."
#prompt = f"{judge_prompt} {message_prompt}"
#output = ask_llama4(prompt)
#print("Risposta API:")
#print(output)


