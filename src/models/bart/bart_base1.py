from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import evaluate
import numpy as np

# 1. Carica dataset
dataset = load_dataset("json", data_files="icdar_eng_monograph.jsonl", split="train")

# 2. Tokenizzatore e modello
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 3. Preprocessing
max_input_length = 512
max_target_length = 128

def preprocess(example):
    model_input = tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_dataset = dataset.map(preprocess, batched=False)

# 4. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 5. Metriche
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

# 6. Training args
training_args = TrainingArguments(
    output_dir="./bart-postocr",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # se usi GPU moderna
    logging_steps=100,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Train!
trainer.train()

# 9. Salva il modello
trainer.save_model("bart-postocr-finetuned")
