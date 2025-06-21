from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import re 
import os
import time
import json


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === EVALUATION FUNCTION ===
def valuta_judge(text,model,tokenizer, JUDGE_PROMPT, MAX_NEW_TOKENS, device):
    prompt = (
        f"[PROMPT]\nCorreggi: {text['ocr']}\n\n"
        f"[EXPECTED]\n{text['gold']}\n\n"
        f"[GENERATED]\n{text['model_correction']}\n\n"
        f"{JUDGE_PROMPT}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Stampa tutto ciò che ha risposto Prometheus
    print("\n--- OUTPUT GREZZO DEL MODELLO ---")
    print(decoded)
    print("--- FINE OUTPUT ---\n")

    # Estrai il numero se presente
    match = re.search(r"\[NUMERIC SCORE\]\s*[:\-]?\s*([1-5])\b", decoded)
    return match.group(1) if match else f"[ERRORE: {decoded.strip()}]"

# === LOADING PROMETHEUS MODEL ===
def init():
    MODEL_PATH = os.path.abspath("./src/models/prometheus/cache/models--Unbabel--M-Prometheus-7B/snapshots/030fb74806e4228c466a98706a297d43b31ce5df")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    JUDGE_PROMPT = (
        "Evaluate the quality of the [GENERATED] text in comparison to the [EXPECTED] text. "
        "Use the following scale:\n\n"
        "1 = Completely unacceptable. The output is severely incomplete or entirely incorrect.\n"
        "2 = Major issues. The output is complete but significantly inaccurate or misleading.\n"
        "3 = Some errors. Mostly correct, but with noticeable mistakes or altered meaning.\n"
        "4 = Minor issues. Largely accurate, with only small grammatical or factual deviations.\n"
        "5 = Perfect. No errors; completely faithful and correct.\n\n"
        "Output only the score after the [NUMERIC SCORE] tag.\n"
        "Format: [NUMERIC SCORE] <number>\n"
        "Do not explain. Do not write anything else."
    )
    MAX_NEW_TOKENS = 2500
    log("Caricamento Prometheus base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
    model.eval()

    return MODEL_PATH, device, JUDGE_PROMPT, MAX_NEW_TOKENS, tokenizer, model




def prometheus_ask_score(original,reference,correction, JUDGE_PROMPT, MAX_NEW_TOKENS, device, tokenizer, model):
    text = {
            "ocr": original,
            "gold": reference,
            "model_correction": correction,
        }

    # === LOOP DI VALUTAZIONE ===
    log("Valutazione esempi...")
    score = valuta_judge(text,model,tokenizer, JUDGE_PROMPT, MAX_NEW_TOKENS, device)
    print(f"OCR:               {original}")
    print(f"Gold:              {reference}")
    print(f"Model Correction:  {correction}")
    print(f"Score:             {score}")

    return score



def prometheus_score(FILE_NAME, correction_model):
    MODEL_PATH, device, JUDGE_PROMPT, MAX_NEW_TOKENS, tokenizer, model = init()

    BASE_PATH = "datasets/eng/corrections/" + correction_model + "/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    # Carica il tuo JSON da file
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data = list(data.values())

    key = f"correction"
    key2 = f"prometheus_score"



    log("|====================================")
    log(f"Evaluating with Prometheus...")
    for i, entry in enumerate(data, start=1):
        
        correction = entry[key]
        reference = entry["gold"]
        original = entry["ocr"]

        try:
            score = prometheus_ask_score(original,reference,correction, JUDGE_PROMPT, MAX_NEW_TOKENS, device, tokenizer, model)
            entry[key2] = int(score)
        except ValueError:
            entry[key2] = score  
        except Exception as e:
            log(f"Errore alla voce {i}: {e}")
            entry[key2] = "ERROR"

        time.sleep(1) 

    # Salva il risultato in un nuovo file
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log("|====================================")
    log("\n")