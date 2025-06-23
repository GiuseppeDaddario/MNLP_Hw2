from accelerate import Accelerator
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
    return match.group(1) if match else f"[ERRORE: {decoded.strip()}]", decoded

# === LOADING PROMETHEUS MODEL ===
def init():
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    HF_MODEL_NAME = "Unbabel/M-Prometheus-7B"
    LOCAL_MODEL_PATH = os.path.abspath("./src/models/prometheus/cache/models--Unbabel--M-Prometheus-7B/snapshots/030fb74806e4228c466a98706a297d43b31ce5df")

    if is_colab:
        MODEL_PATH = HF_MODEL_NAME
        local_files_only = False
        log("Running in Google Colab")
    else:
        MODEL_PATH = LOCAL_MODEL_PATH
        local_files_only = True
        log("Running in Local environment")

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
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=local_files_only)
    model = accelerator.prepare(model)
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
    score, full_output = valuta_judge(text,model,tokenizer, JUDGE_PROMPT, MAX_NEW_TOKENS, device)
    print(f"OCR:               {original}")
    print(f"Gold:              {reference}")
    print(f"Model Correction:  {correction}")
    print(f"Score:             {score}")

    return score, full_output



def prometheus_score(FILE_NAME, correction_model):
    MODEL_PATH, device, JUDGE_PROMPT, MAX_NEW_TOKENS, tokenizer, model = init()

    groupname = "C0rr3tt0r1_4ut0m4t1c1"
    evaluation_model = "prometheus"
    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    JUDGE_PATH = f"outputs/{correction_model}/{groupname}-hw2_ocr-{evaluation_model}.json"

    # Carica JSON da file
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #data = list(data.values())
    key = "correction"
    key2 = "prometheus_score"

    judge_output_data = {}

    log("|====================================")
    log(f"\033[34mEvaluating with Prometheus...\033[0m")


    for i, entry in enumerate(data, start=1):
        correction = entry[key]
        reference = entry["gold"]
        original = entry["ocr"]

        try:
            score, full_output = prometheus_ask_score(original, reference, correction, JUDGE_PROMPT, MAX_NEW_TOKENS, device, tokenizer, model)
            entry[key2] = int(score)
            judge_output_data[str(i-1)] = {
                "output": full_output,
                "score": score
            }
        except ValueError:
            score = entry[key2] = score
        except Exception as e:
            log(f"Errore alla voce {i}: {e}")
            score = entry[key2] = "ERROR"


    # Salva il risultato con gli score nel file originale
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Salva anche il nuovo file richiesto (JUDGE_PATH)
    os.makedirs(os.path.dirname(JUDGE_PATH), exist_ok=True)
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(judge_output_data, f, ensure_ascii=False, indent=2)

    log(f"| \033[32mSalvato file di valutazione: {JUDGE_PATH}\033[0m")
    log("|====================================")
    log("\n")