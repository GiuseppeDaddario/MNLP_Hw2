from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import json

# 1. Carica dataset JSON e converti in Dataset HuggingFace
with open("extra_stuff\datasets\human_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Se il dataset è lista di dict:
dataset = Dataset.from_list(data)

# 2. Tokenizer e modello
model_name = "DeepMount00/OCR_corrector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Tokenizzazione (con padding e truncation)
def preprocess_function(examples):
    inputs = ["fix: " + ocr for ocr in examples["ocr"]]
    targets = examples["corretto"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Definisci argomenti di training
training_args = Seq2SeqTrainingArguments(
    output_dir="./ocr_corrector_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,  # o True se hai GPU e supporta FP16
)

# 5. Crea Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 6. Fine-tuning
trainer.train()

# 7. Salva modello fine-tunato
trainer.save_model("./ocr_corrector_finetuned")
