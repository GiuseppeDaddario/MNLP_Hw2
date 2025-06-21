#pretrained model for our task (JUST ITALIAN!!)


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "DeepMount00/OCR_corrector"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(device)
my_text = "povero  Pinoccliio,  che  aveva  sempre  gii  oc- \nchi fra  il  sonno,  non  s'era  ancora  avvisto  dei \npiedi  che  gli  si  erano  tutti  brnciati:"
inputs = tokenizer(my_text, return_tensors="pt").to(device)
outputs = model.generate(input_ids=inputs['input_ids'],
               attention_mask=inputs['attention_mask'],
               num_beams=2, max_length=1050, top_k=10)
clean_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(clean_text)