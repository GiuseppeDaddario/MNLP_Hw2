from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import json

# Load custom dataset
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)  # carica tutta la lista di dizionari


BASE_PATH = "/leonardo/home/userexternal/lbenucci/MNLP_Hw2/datasets/eng/"

data = load_json(BASE_PATH + "finetuning.json")
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing
max_input_length = 128
max_target_length = 128

def preprocess(example):
    inputs = tokenizer(example["ocr"], max_length=max_input_length, truncation=True, padding="max_length")
    targets = tokenizer(example["corretto"], max_length=max_target_length, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_ds = dataset.map(preprocess, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training config
training_args = TrainingArguments(
    output_dir="./t5-ocr-correction",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    save_strategy="epoch",
    fp16=True,  # usa fp16 con A100
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model
trainer.save_model("./t5-ocr-correction-final")
tokenizer.save_pretrained("./t5-ocr-correction-final")
