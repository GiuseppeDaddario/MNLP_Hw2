# bart_inference.py

from transformers import BartTokenizer, BartForConditionalGeneration

# Percorso del modello salvato
model_path = "./bart_results/best_checkpoint"

# Carica tokenizer e modello
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Frase OCR da correggere
ocr_input = "per  sal- \nvarsi, \u00e8  ingoiato  dal  terribile  Pesce-cane. \nDopo  cinquanta  minuti  clie  il  duellino  era \nsott'acqua^  il  compratore  disse"

# Preprocessing
input_ids = tokenizer(ocr_input, return_tensors="pt", max_length=128, truncation=True).input_ids

# Generazione
outputs = model.generate(input_ids, num_beams=4, max_length=128, early_stopping=True)

# Output corretto
corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🔹 OCR Input: ", ocr_input)
print("✅ Output:    ", corrected)
