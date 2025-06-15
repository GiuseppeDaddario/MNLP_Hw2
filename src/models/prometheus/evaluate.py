from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import re 

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === CONFIG ===
MODEL_PATH = "cache/models--Unbabel--M-Prometheus-7B/snapshots/030fb74806e4228c466a98706a297d43b31ce5df"
device = "cuda" if torch.cuda.is_available() else "cpu"

JUDGE_PROMPT = (
    "Evaluate the quality of the [GENERATED] text in comparison to the [EXPECTED] text. "
    "Use the following scale:\n\n"
    "1 = Completely unacceptable\n"
    "2 = Severe errors\n"
    "3 = Partially wrong, mostly minor errors\n"
    "4 = Good, but not perfect\n"
    "5 = Perfect output\n\n"
    "Only output the score **after** the [NUMERIC SCORE] tag.\n"
    "Format: [NUMERIC SCORE] <number>\n"
    "Do not explain. Do not write anything else."
)

# === ESEMPI ===
examples = [
    {
        "ocr": "Tntroduzìone aìì'ìntellìgenza artìficìale",
        "corretto": "Introduzione all'Intelligenza Artificiale",
        "output_generato": "Introduzione all’Intelligenza Artificiale"
    },
    {
        "ocr": "Universta' dgli Studi di Roma ‘La Spieenza’",
        "corretto": "Università degli Studi di Roma 'La Sapienza'",
        "output_generato": "Universita' degli Studi di Roma “La Sapienza”"
    },
]

# === CARICAMENTO MODELLO ===
log("Caricamento Prometheus base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
model.eval()

# === FUNZIONE DI VALUTAZIONE ===
def valuta_judge(example):
    prompt = (
        f"[PROMPT]\nCorreggi: {example['ocr']}\n\n"
        f"[EXPECTED]\n{example['corretto']}\n\n"
        f"[GENERATED]\n{example['output_generato']}\n\n"
        f"{JUDGE_PROMPT}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔍 Stampa tutto ciò che ha risposto Prometheus
    print("\n--- OUTPUT GREZZO DEL MODELLO ---")
    print(decoded)
    print("--- FINE OUTPUT ---\n")

    # Estrai il numero se presente
    match = re.search(r"\[NUMERIC SCORE\]\s*([1-5])", decoded)
    return match.group(1) if match else f"[ERRORE: {decoded.strip()}]"


# === LOOP DI VALUTAZIONE ===
log("Valutazione esempi...")
for i, ex in enumerate(examples, 1):
    score = valuta_judge(ex)
    print(f"\n--- ESEMPIO {i} ---")
    print(f"OCR:       {ex['ocr']}")
    print(f"Corretto:  {ex['corretto']}")
    print(f"Generato:  {ex['output_generato']}")
    print(f"Valutazione: {score}")