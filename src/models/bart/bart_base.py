# train_bart_ocr.py





import json
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments



ds = load_dataset("PleIAs/Post-OCR-Correction", "italian")
print(ds["train"][0])




# 1. Carica il tuo dataset OCR (assunto JSON con "ocr" e "correct")
with open("datasets/finetuning_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)
samples = [{"ocr": d["ocr"], "correct": d["corretto"]} for d in data if d["ocr"] and d["corretto"]]

# 2. Split train/test
train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)

# 3. Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# 4. Preprocessing
def preprocess(example):
    inputs = tokenizer(example["ocr"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example["correct"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# 5. Dataset HF
train_dataset = Dataset.from_list(train_data).map(preprocess, remove_columns=["ocr", "correct"])
test_dataset = Dataset.from_list(test_data).map(preprocess, remove_columns=["ocr", "correct"])

# 6. Model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 7. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,  # Usa fp16 se hai GPU che lo supporta
    save_total_limit=2,
    load_best_model_at_end=True,
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 9. Train & Save
trainer.train()
trainer.save_model("./bart_results/best_checkpoint")
