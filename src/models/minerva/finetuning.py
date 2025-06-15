from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from functools import partial

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === Config ===
MODEL_PATH = "./minerva-cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
DATA_PATH = "./datasets/finetuning_all.json"
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-5

# === Init accelerator ===
log("Initializing the accelerator...")
accelerator = Accelerator(mixed_precision="fp16")

# === Load model/tokenizer ===
log("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

# === Fix missing pad token ===
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LoRA config ===
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

# === Load dataset ===
log("Loading finetuning dataset...")
raw_dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def preprocess(example, tokenizer):
    prompt = f"Correggi: {example['ocr']}\nRisposta:"
    target = example["corretto"]
    full_text = f"{prompt} {target}"
    return tokenizer(full_text, truncation=True, max_length=512)

# Pre-tokenize to avoid doing it at batch time
tokenized_dataset = raw_dataset.map(
    lambda ex: preprocess(ex, tokenizer),
    remove_columns=raw_dataset.column_names
)

# === Collate function ===
def smart_collate(batch, tokenizer):
    padded = tokenizer.pad(batch, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()
    return padded

collate_fn = partial(smart_collate, tokenizer=tokenizer)

# === Dataloader ===
log("Configuring dataloader")
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# === Optimizer and scheduler ===
log("Setting optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# === Prepare everything with accelerator ===
log("Preparing everything...")
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# === Training loop ===
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
    log(f"Epoch {epoch+1} finished. Avg loss: {total_loss / len(train_dataloader):.4f}")

# === Save model/tokenizer ===
log("Saving model...")
if accelerator.is_main_process:
    model.save_pretrained("./results_minerva_ocr")
    tokenizer.save_pretrained("./results_minerva_ocr")