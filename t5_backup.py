
### QUESTO QUA SEMBRA FUNZIONICCHIARE




import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


# 1. Carica il dataset
with open("datasets/eng/finetuning.json", "r", encoding="utf-8") as f:
    data = json.load(f)

samples = [{"ocr": item["ocr"], "correct": item["corretto"]} for item in data if item["ocr"] and item["corretto"]]

# 2. Split 80/20
train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)

# 3. Tokenizer T5
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 4. Preprocessing
def preprocess(example):
    input_text = "fix: " + example["ocr"]  # prefix task instruction
    target_text = example["correct"]
    inputs = tokenizer(input_text, max_length=256, padding="max_length", truncation=True) #default 256
    targets = tokenizer(target_text, max_length=256, padding="max_length", truncation=True) #default 256
    inputs["labels"] = targets["input_ids"]
    return inputs

# 5. Create Hugging Face Dataset
train_dataset = Dataset.from_list(train_data).map(preprocess, remove_columns=["ocr", "correct"])
test_dataset = Dataset.from_list(test_data).map(preprocess, remove_columns=["ocr", "correct"])

# 6. Load model
model = T5ForConditionalGeneration.from_pretrained("t5-base")


# 7. TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True  # ora è valido qui!
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 9. Train
trainer.train()

# 10. Evaluate on test set
outputs = trainer.predict(test_dataset)
decoded_preds = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(outputs.label_ids, skip_special_tokens=True)

# 11. Print some examples
for i in range(5):
    print(f"\n🔹 OCR:     {test_data[i]['ocr']}")
    print(f"✅ Truth:   {test_data[i]['correct']}")
    print(f"🤖 T5 Out:  {decoded_preds[i]}")


