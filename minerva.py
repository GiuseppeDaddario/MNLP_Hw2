import transformers
import torch
from src.hf_login import HF_Login


HF_Login()

model_id = "sapienzanlp/Minerva-3B-base-v1.0"

# Initialize the pipeline.
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Input text for the model.

prompt = "Correggi solo la frase seguente senza aggiungere altro:\nil c4ne è un 4nimale domestico\nCorrezione:"

output = pipeline(
    prompt,
    max_new_tokens=20,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
)

print(output[0]['generated_text'])

