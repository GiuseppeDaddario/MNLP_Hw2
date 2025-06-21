#pretrained model for our task (JUST ITALIAN!!)


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "DeepMount00/OCR_corrector"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(device)
my_text = "THEsuperstition upon which this taIe iſ founded is very general in the East. Among tho Arabjans itappeors to be common: it did not, however, extend itself to the Greeks until after the establi shment of Christianity; and it has only aſsumed its prosent form since the division af the Latin and Greok churches; at which time, lhe idea becoming prevalent, that a Lcltin body could not corrvpl if buried in their territory, it gradually increosed, and formed lhe subject of many wonderful stories, ſtill extant, of the dead rising from their graves, and feeding uponlhe blood of tho young and beautiful. In the West itspread, with some slight variation, all over Hungary, Poland, Austria, and Lorraine, whoro the helies existed, that vompyresnightly imbi6ed a certain portion of the blood of their victims, who became emaciated, lost their strength, and speedily died of c0nsumptions; whilst these human blood-suckers fattened—and their veins became distended to such a state of ropletion, as t0 cause the blood to flow from all the passages of their badies, and even fr0m the ucry pores of thoir skins."
inputs = tokenizer(my_text, return_tensors="pt").to(device)
outputs = model.generate(input_ids=inputs['input_ids'],
               attention_mask=inputs['attention_mask'],
               num_beams=2, max_length=1050, top_k=10)
clean_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(clean_text)