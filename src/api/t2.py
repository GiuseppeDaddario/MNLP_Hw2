import json
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from transformers import DataCollatorWithPadding
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==== 1. Dataset Personalizzato ====

class OcrCorrectionDataset(Dataset):
    def __init__(self, dirty_path, clean_path, tokenizer, max_length=128):
        with open(dirty_path, 'r', encoding='utf-8') as f:
            self.dirty = json.load(f)
        with open(clean_path, 'r', encoding='utf-8') as f:
            self.clean = json.load(f)

        self.keys = sorted(self.dirty.keys(), key=lambda x: int(x))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        dirty_text = "fix: " + self.dirty[key]
        clean_text = self.clean[key]

        inputs = self.tokenizer(dirty_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(clean_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        labels = targets.input_ids.squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignora il padding nel loss

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels
        }


# ==== 2. Setup ====

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# Percorsi dei tuoi file
train_dirty_path = "datasets/eng/the_vampyre_ocr.json"
train_clean_path = "datasets/eng/the_vampyre_clean.json"

dataset = OcrCorrectionDataset(train_dirty_path, train_clean_path, tokenizer)

# ==== 3. Training ====

training_args = TrainingArguments(
    output_dir="./t5-ocr-correction",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",  # puoi cambiare in "epoch" se hai val set
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()




################################ EVALUATION ################################


# ==== 4. Test interattivo (post-training) ====

def correct_text(model, tokenizer, input_text):
    input_ids = tokenizer("fix: " + input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Esempio
model.eval()
example = "This iss jusst a text and sonne err0rs 1n the phnase."
print("Corretto:", correct_text(model, tokenizer, example))
