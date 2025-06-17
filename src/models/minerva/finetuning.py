from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from functools import partial

# === Logging ===
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === Config ===
MODEL_PATH = "./src/models/minerva/cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
BATCH_SIZE = 3
EPOCHS = 4
LR = 2e-5
FINETUNED_MODEL_PATH = "./src/models/minerva/finetuned_minerva_all"

# === Funzioni per caricare e formattare i dataset ===
def load_ocr_dataset(path):
    ds = load_dataset("json", data_files=path)["train"]
    def format_ocr(example):
        prompt = (
            "You must correct the following OCR sentence, replying only with the clean sentence "
            "keeping syntax, language and meaning.\n"
            f"Sentence to correct: {example['ocr']}\nYour Answer:"
        )
        return {
            "instruction": prompt,
            "response": example["corretto"]
        }
    return ds.map(format_ocr, remove_columns=ds.column_names)

def load_lima_dataset(path):
    ds = load_from_disk(path)["train"]
    ds = ds.filter(lambda ex: len(ex["conversations"]) >= 2)
    def format_lima(example):
        return {
            "instruction": example["conversations"][0],
            "response": example["conversations"][1]
        }
    return ds.map(format_lima, remove_columns=ds.column_names)

# === Preprocessing ===
def preprocess(example, tokenizer):
    full_text = f"### Istruzione:\n{example['instruction']}\n\n### Risposta:\n{example['response']}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

# === Collate ===
def smart_collate(batch, tokenizer):
    padded = tokenizer.pad(batch, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()
    return padded

# === Accelerate ===
log("Initializing the accelerator...")
accelerator = Accelerator(mixed_precision="fp16")

# === Model & Tokenizer ===
log("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# === Pad token fix ===
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LoRA ===
log("Configuring LoRA...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# === Dataset ===
log("Loading and combining datasets...")
ds1 = load_ocr_dataset("./datasets/eng/finetuning.json")
ds2 = load_ocr_dataset("./datasets/eng/human_data.json")
ds3 = load_lima_dataset("./datasets/lima")
combined_dataset = concatenate_datasets([ds1, ds2, ds3])

# === Tokenization ===
log("Tokenizing dataset...")
tokenized_dataset = combined_dataset.map(
    lambda ex: preprocess(ex, tokenizer),
    remove_columns=combined_dataset.column_names
)

# === DataLoader ===
log("Configuring dataloader...")
collate_fn = partial(smart_collate, tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# === Optimizer & Scheduler ===
log("Setting optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# === Accelerator Prepare ===
log("Preparing with accelerator...")
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# === Training ===
log("Start training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    log(f"Epoch {epoch+1}/{EPOCHS} finished. Avg loss: {total_loss / len(train_dataloader):.4f}")

# === Save model ===
log("Saving model...")
if accelerator.is_main_process:
    model.save_pretrained(FINETUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINETUNED_MODEL_PATH)