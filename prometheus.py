from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Nome del modello su Hugging Face
model_name = "prometheus-eval/prometheus-7b-v2.0"

# Carica tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Funzione per generare testo
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Esempio di prompt
prompt = "Cia0, come st4i oggi?"
print(generate_text(prompt))