import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, Dataset

# === Imposta directory offline ===
MODEL_PATH = "/leonardo/models/Minerva-7B-instruct-v1.0"
DATA_PATH = "/leonardo/data/ocr_dataset.json"

# === Carica modello e tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# === Configura LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# === Carica dataset ===
raw_dataset = load_dataset("json", data_files=DATA_PATH)["train"]

# === Preprocessing ===
def tokenize(example):
    prompt = f"Correggi: {example['ocr']}\nRisposta:"
    target = example["corretto"]
    full_text = f"{prompt} {target}"
    tokens = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

# === Argomenti di training ===
training_args = TrainingArguments(
    output_dir="./results_minerva_ocr",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# === Trainer ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Avvio training ===
trainer.train()
model.save_pretrained("./results_minerva_ocr")
tokenizer.save_pretrained("./results_minerva_ocr")