from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Percorso al modello fine-tunato
MODEL_PATH = "bart-postocr-finetuned"

# Carica tokenizer e modello
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()  # modalità inferenza

# Se hai GPU, usa .to("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === FUNZIONE DI INFERENZA ===
def correct_text(ocr_text, max_length=128):
    inputs = tokenizer(ocr_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === ESEMPI ===
ocr_examples = [
    "contcmplate them w1th interest for hours hc has contrived to g1ve St. John an aImost perfect expresslon of 'div1ne philosopHy'",
    "the strugglc betwcen the poetry of the heart and the opposîng prose of outward cîrcumstances"
]

for ocr in ocr_examples:
    corrected = correct_text(ocr)
    print(f"\nOCR   : {ocr}")
    print(f"Clean : {corrected}")
