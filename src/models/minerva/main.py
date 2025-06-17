import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.prometheus import evaluate

# === Config ===
BASE_MODEL_PATH = "./src/models/minerva/cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
#BASE_MODEL_PATH = "./src/models/minerva/finetuned_minerva"
FINETUNED_MODEL_PATH = "./src/models/minerva/finetuned_minerva_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === Load models ===
def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
# === Prompt ===
def make_prompt(ocr_text):
    return f"You must correct the following OCR sentence, replying only with the clean sentence keeping syntax, language and meaning.\nSentence to correct: {ocr_text}\nYour Answer:"

# === Funzione di generazione pulita ===
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    max_new_tokens = 4096
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        return_dict_in_generate=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Your Answer:" in decoded:
        text = decoded.split("Your Answer:")[-1].strip()
    else:
        text = decoded.strip()

    # Prendi solo fino al primo a capo
    first_line = text.split('\n')[0].strip()
    return first_line


# === Load tokenizer ===
log("Loading tokenizer...")
tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer_finetuned = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)

if tokenizer_base.pad_token is None:
    tokenizer_base.pad_token = tokenizer_base.eos_token

if tokenizer_finetuned.pad_token is None:
    tokenizer_finetuned.pad_token = tokenizer_finetuned.eos_token

log("Loading base Minerva...")
model_base = load_model(BASE_MODEL_PATH)
log("Loading finetuned Minerva...")
model_finetuned = load_model(FINETUNED_MODEL_PATH)

# === Esempi OCR ===
ocr_examples = [
    	"THEsuperstition upon which this taIe iſ founded is very general in the East. Among tho Arabjans itappeors to be common: it did not, however, extend itself to the Greeks until after the establi shment of Christianity; and it has only aſsumed its prosent form since the division af the Latin and Greok churches; at which time, lhe idea becoming prevalent, that a Lcltin body could not corrvpl if buried in their territory, it gradually increosed, and formed lhe subject of many wonderful stories, ſtill extant, of the dead rising from their graves, and feeding uponlhe blood of tho young and beautiful. In the West itspread, with some slight variation, all over Hungary, Poland, Austria, and Lorraine, whoro the helies existed, that vompyresnightly imbi6ed a certain portion of the blood of their victims, who became emaciated, lost their strength, and speedily died of c0nsumptions; whilst these human blood-suckers fattened—and their veins became distended to such a state of ropletion, as t0 cause the blood to flow from all the passages of their badies, and even fr0m the ucry pores of thoir skins."
]
clean = "THE superstition upon which this tale is founded is very general in the East. Among the Arabians it appears to be common: it did not, however, extend itself to the Greeks until after the establishment of Christianity; and it has only assumed its present form since the division of the Latin and Greek churches; at which time, the idea becoming prevalent, that a Latin body could not corrupt if buried in their territory, it gradually increased, and formed the subject of many wonderful stories, still extant, of the dead rising from their graves, and feeding upon the blood of the young and beautiful. In the West it spread, with some slight variation, all over Hungary, Poland, Austria, and Lorraine, where the belief existed, that vampyres nightly imbibed a certain portion of the blood of their victims, who became emaciated, lost their strength, and speedily died of consumptions; whilst these human blood-suckers fattened—and their veins became distended to such a state of repletion, as to cause the blood to flow from all the passages of their bodies, and even from the very pores of their skins."


# === Comparazione ===
log("\n=== CONFRONTO MINERVA BASE vs FINE-TUNED ===\n")
for i, ocr in enumerate(ocr_examples):
    prompt = make_prompt(ocr)
    base_output = generate(model_base, tokenizer_base, prompt)
    fine_output = generate(model_finetuned, tokenizer_finetuned, prompt)

    print(f"--- ESEMPIO {i+1} ---")
    print(f"OCR:         {ocr}")
    print(f"Base:        {base_output}")
    print(f"Fine-tuned:  {fine_output}")
    print()

texts = [
            {
                "ocr": f"{ocr}",
                "correct": f"{clean}",
                "generation": f"{base_output}"
            },
            {
                "ocr": f"{ocr}",
                "correct": f"{clean}",
                "generation": f"{fine_output}"
            }
        ]

evaluate(texts)