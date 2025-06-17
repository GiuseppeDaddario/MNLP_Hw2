from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# === Imposta il percorso del modello fine-tunato ===
MODEL_PATH = "./t5-ocr-correction-final"  # o un path assoluto

# === Carica modello e tokenizer ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)


# === Funzione per correggere il testo ===
def correct_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,
        num_beams=5,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Esempio di uso ===
if __name__ == "__main__":
    test_samples = [
        "Quesfo é un esempi0 di t3sto con err0ri.",
        "Th3 quiqk br0wn fox jmps ov3r t3h la2y d0g.",
        "L0rem ip$um do1or sit am3t."
    ]

    for noisy in test_samples:
        corrected = correct_text(noisy)
        print(f"\nOCR:      {noisy}")
        print(f"Corretto: {corrected}")
