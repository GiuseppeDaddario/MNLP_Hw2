from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from functools import partial

# --- Logging ---
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- Config ---
MODEL_PATH = "./src/models/minerva/cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
FINETUNED_MODEL_PATH = "./src/models/minerva/finetuned_minerva_llima"
BATCH_SIZE = 3
EPOCHS = 4
LR = 2e-5
MAX_LENGTH = 512

# --- Prompt template ---
def make_prompt(ocr_text: str) -> str:
    return (
        "You are an OCR correction system.\n"
        "Task: Fix spelling, spacing, and OCR errors.\n"
        "Rules:\n"
        "1. Do NOT explain.\n"
        "2. Keep old spellings and historical terms.\n"
        "3. No additions.\n"
        "4. If already correct, repeat it.\n"
        f"Sentence: {ocr_text}\n"
        "Corrected:"
    )

# --- Dataset loaders ---
def load_ocr_dataset(path):
    ds = load_dataset("json", data_files=path)["train"]
    return ds.map(lambda ex: {"instruction": make_prompt(ex["ocr"]), "response": ex["corretto"]}, remove_columns=ds.column_names)

def load_lima_dataset(path):
    ds = load_from_disk(path)["train"]
    ds = ds.filter(lambda ex: len(ex["conversations"]) >= 2)
    return ds.map(lambda ex: {"instruction": ex["conversations"][0], "response": ex["conversations"][1]}, remove_columns=ds.column_names)

# --- Preprocessing ---
def preprocess(example, tokenizer):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc["labels"] = [lbl if mask else -100
                     for lbl, mask in zip(enc["input_ids"], enc["attention_mask"])]
    return enc

# --- Collate fn ---
def smart_collate(batch, tokenizer):
    padded = tokenizer.pad(batch, return_tensors="pt")
    labels = padded["labels"]
    padded["labels"] = torch.where(padded["attention_mask"] == 1, labels, -100)
    return padded

# --- Accelerator init ---
accelerator = Accelerator(mixed_precision="fp16")
log("Accelerator ready.")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# --- LoRA config ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]  # controlla che siano presenti nel tuo modello
)
model = get_peft_model(model, peft_config)

# --- Load and prepare datasets ---
#ds1 = load_ocr_dataset("./datasets/eng/finetuning.json")
#ds2 = load_ocr_dataset("./datasets/eng/human_data.json")
ds3 = load_lima_dataset("./datasets/lima")
#combined = concatenate_datasets([ds1, ds2, ds3])
combined = ds3

tokenized = combined.map(lambda ex: preprocess(ex, tokenizer), batched=True)
train_dl = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial(smart_collate, tokenizer=tokenizer))

# --- Optimizer & scheduler ---
optimizer = AdamW(model.parameters(), lr=LR)
num_steps = EPOCHS * len(train_dl)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)

# --- Prepare everything with accelerator ---
model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)

# --- Training loop ---
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    log(f"Epoch {epoch} completed — avg loss: {total_loss/len(train_dl):.4f}")

# --- Save LoRA adapter + tokenizer ---
if accelerator.is_main_process:
    model.save_pretrained(FINETUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINETUNED_MODEL_PATH)

log("✅ Finetuning completed.")