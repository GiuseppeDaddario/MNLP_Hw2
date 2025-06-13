import torch
from torch import nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

class t5Model(nn.Module):
    def __init__(self, model_name="t5-base", max_length=128):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.max_length = max_length

    def forward(self, input_texts):
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        inputs = self.tokenizer(
            ["fix: " + text for text in input_texts],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length
            )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def fit(self,train_dataset,output_dir="./t5-ocr-correction",batch_size=8,num_epochs=10,learning_rate=3e-4,weight_decay=0.01,logging_steps=10):

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_strategy="epoch",
            eval_strategy="no",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        trainer.train()

