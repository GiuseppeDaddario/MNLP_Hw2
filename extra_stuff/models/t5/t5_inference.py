import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carica tokenizer base e modello fine-tuned
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5_results/checkpoint-160")

# Frase OCR da correggere
ocr_input = "The gvardians hastened to protect Miss Aubrey; but wben they arrivcd, it was too 1 ate."

# Prepara l'input
input_text = "fix: " + ocr_input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)

# Inferenza
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )

# Decodifica output
corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Stampa
print(f"\n🔹 OCR Input:  {ocr_input}")
print(f"✅ Output:     {corrected_text}")
