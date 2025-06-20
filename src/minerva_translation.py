
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json
import os
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BASE_MODEL_PATH = "./src/models/minerva/cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
FINETUNED_MODEL_PATH = "./src/models/minerva/finetuned_minerva_all"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_minerva_model(finetuned=True):
    model_path = FINETUNED_MODEL_PATH if finetuned else BASE_MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer

def make_prompt(ocr_text: str) -> str:
    return (
        "You are an OCR correction system.\n"
        "Task: Fix spelling, spacing, and OCR errors in the given sentence ONLY if there are any.\n"
        "Rules:\n"
        "1. DO NOT explain.\n"
        "2. DO NOT change names, old spellings or historical terms.\n"
        "3. DO NOT add any text.\n"
        "4. If the sentence is already correct, repeat it exactly.\n"
        "Sentence: " + ocr_text + "\n"
        "Corrected:"
    )

@torch.inference_mode()
def ask_minerva(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Corrected:" in decoded:
        after_corrected = decoded.split("Corrected:", 1)[-1].strip()
        first_line = after_corrected.split("\n")[0].strip()
        return first_line, decoded

    return decoded.strip().split("\n")[0].strip(), decoded


def split_into_sentences(text):
    # Divide le frasi usando il punto seguito da spazio o fine stringa
    return [s.strip() for s in re.split(r'(?<=[.])\s+', text) if s.strip()]

def process_ocr_file(input_file, gold_file, output_file, model, tokenizer, force_indices=None):
    with open(input_file, "r", encoding="utf-8") as f_in:
        ocr_data = json.load(f_in)

    with open(gold_file, "r", encoding="utf-8") as f_gold:
        gold_data = json.load(f_gold)

    # Carica output esistente, se disponibile
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_out:
            results = json.load(f_out)
    else:
        results = {}

    keys = list(ocr_data.keys())

    # Applica filtro su indici forzati
    if force_indices is not None:
        keys = [keys[i] for i in force_indices if i < len(keys)]

    for i, key in enumerate(keys):
        # Salta se già processato (solo se non forzato)
        if key in results and (force_indices is None or i not in force_indices):
            log(f"[{i+1}/{len(keys)}] Skipping already processed key '{key}'")
            continue

        ocr_text = ocr_data.get(key, "").strip()
        if not ocr_text:
            continue

        gold_text = gold_data.get(key, "").strip()
        sentences = split_into_sentences(ocr_text)
        corrected_sentences = []

        log(f"[{i+1}/{len(keys)}] Processing key '{key}'... ({len(sentences)} sentences)")
        
        for j, sentence in enumerate(sentences):
            if not sentence:
                continue
            prompt = make_prompt(sentence)
            corrected, full_text = ask_minerva(prompt, model, tokenizer)
            corrected_sentences.append(corrected)

            print(f"------------ DEBUG [{j+1}/{len(sentences)}] --------------")
            print(f"{full_text}")
            print("------------------------------------------")

        final_correction = " ".join(corrected_sentences)

        log(f"📝 OCR:        {ocr_text}")
        log(f"🤖 Correction: {final_correction}")
        log(f"✅ Gold:       {gold_text}\n")

        results[key] = {
            "ocr": ocr_text,
            "gold": gold_text,
            "minerva_correction": final_correction
        }

        # Salva dopo ogni iterazione per sicurezza
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)

def translate_with_minerva(file_name):
    print("\n")
    print("|========================================")
    print("| \033[93mTranslating with minerva ...\033[0m")
    
    input_path = f"datasets/eng/{file_name}_ocr.json"
    gold_path = f"datasets/eng/{file_name}_clean.json"
    output_path = f"datasets/eng/corrections/{file_name}.json"


    log("Loading Minerva model...")
    model, tokenizer = load_minerva_model(finetuned=True)

    log("Starting OCR correction...")
    process_ocr_file(input_path, gold_path, output_path, model, tokenizer,force_indices=None)

    log("✅ Done!")
