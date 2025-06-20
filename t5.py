import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split


# Funzione per splittare OCR e corretto in segmenti max_len mantenendo l’allineamento
def split_example(ocr_text, correct_text, max_len=50):
    split_ocr = []
    split_correct = []
    i = 0
    while i < len(ocr_text):
        end = i + max_len
        if end < len(ocr_text):
            space_pos = ocr_text.rfind(" ", i, end)
            if space_pos != -1 and space_pos > i:
                end = space_pos
        else:
            end = len(ocr_text)

        split_ocr.append(ocr_text[i:end].strip())
        split_correct.append(correct_text[i:end].strip())
        i = end
    return list(zip(split_ocr, split_correct))


# 1. Carica il dataset
with open("datasets/eng/finetuning.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Applica lo split per segmenti corti a ogni esempio
samples = []
for item in data:
    ocr = item.get("ocr", "")
    corr = item.get("corretto", "")
    if ocr and corr:
        segments = split_example(ocr, corr, max_len=50)
        for o, c in segments:
            samples.append({"ocr": o, "correct": c})

# 3. Split 80/20
train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)

# 4. Tokenizer T5
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 5. Preprocessing
def preprocess(example):
    input_text = "fix: " + example["ocr"]
    target_text = example["correct"]
    inputs = tokenizer(input_text, max_length=256, padding="max_length", truncation=True)
    targets = tokenizer(target_text, max_length=256, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

# 6. Create Hugging Face Dataset
train_dataset = Dataset.from_list(train_data).map(preprocess, remove_columns=["ocr", "correct"])
test_dataset = Dataset.from_list(test_data).map(preprocess, remove_columns=["ocr", "correct"])

# 7. Load model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 8. TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    fp16=True
)

# 9. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 10. Train
trainer.train()

# 11. Evaluate on test set
outputs = trainer.predict(test_dataset)

import numpy as np

preds = outputs.predictions

# Se preds sono logits (3D), prendi l'argmax sul vocabolario per ottenere gli IDs
if preds.ndim == 3:
    preds = np.argmax(preds, axis=-1)

# Pulisci i token invalidi (es. -100) sostituendoli con pad_token_id
preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(outputs.label_ids, skip_special_tokens=True)

# 12. Print some examples
for i in range(5):
    print(f"\n🔹 OCR:     {test_data[i]['ocr']}")
    print(f"✅ Truth:   {test_data[i]['correct']}")
    print(f"🤖 T5 Out:  {decoded_preds[i]}")
