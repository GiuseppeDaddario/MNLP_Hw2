import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def ask_gpt2(message):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2-xl",
        quantization_config=quantization_config,
        device_map="auto"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    inputs = tokenizer(message, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
