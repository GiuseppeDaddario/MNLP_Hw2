import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Config ===
BASE_MODEL_PATH = "./minerva-cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
FINETUNED_MODEL_PATH = "./results_minerva_ocr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 40

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Load models ===
def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

print("Caricamento modelli...")
model_base = load_model(BASE_MODEL_PATH)
model_finetuned = load_model(FINETUNED_MODEL_PATH)

# === Esempi OCR ===
ocr_examples = [
    "THEsuperstition upon which this taIe iſ founded is very general in the East. Among tho Arabjans itappeors to be common: it did not, however, extend itself to the Greeks until after the establi shment of Christianity; and it has only aſsumed its prosent form since the division af the Latin and Greok churches; at which time, lhe idea becoming prevalent, that a Lcltin body could not corrvpl if buried in their territory, it gradually increosed, and formed lhe subject of many wonderful stories, ſtill extant, of the dead rising from their graves, and feeding uponlhe blood of tho young and beautiful. In the West itspread, with some slight variation, all over Hungary, Poland, Austria, and Lorraine, whoro the helies existed, that vompyresnightly imbi6ed a certain portion of the blood of their victims, who became emaciated, lost their strength, and speedily died of c0nsumptions; whilst these human blood-suckers fattened—and their veins became distended to such a state of ropletion, as t0 cause the blood to flow from all the passages of their badies, and even fr0m the ucry pores of thoir skins.",
    "`VI. \nPinocchio  si  addormenta  coi  piedi  sul  caldano, \ne  la  mattina  dopo  si  sveglia  coi  piedi  tutti  bruciati. \nPer  r  appunto  era  una  nottataccia  d' inferno. \nTonava  forte  forte,  lampeggiava  come  se  il  cielo \npigliasse  fuoco,  e  un  ventaccio  freddo  e  strapaz- \nzone, fischiando  rabbiosamente  e  sollevando  un \nimmenso  nuvolo  di  polvere,  faceva  stridere  e  ci- \ngolare tutti  gli  alberi  della  campagna. \nPinocchio  aveva  una  gran  paura  dei  tuoni  e \ndei  lamj)i  :  se  non  che  la  fame  era  più  forte  della \npaura:  motivo  per  cui  accostò  P uscio  di  casa,  e \npresa  la  carriera,  in  un  centinaio  di  salti  arrivò \nfino  al  paese,  colla  lingua  fuori  e  col  fiato  grosso, \ncome  un  can  da  caccia. \nMa  trovò  tutto  buio  e  tutto  deserto.  Le  bot- \nteghe erano  chiuse;  le  porte  di  casa  chiuse,  le \nfinestre  chiuse,  e  nella  strada  nemmeno  un  cane. \nPareva  il  paese  dei  morti. \nAllora  Pinocchio,  i)reso  dalla  disperazione  e \ndalla  fame,  si  attaccò  al  campanello  d'una  casa,  e \ncominciò  a  sonare  a  distesa,  dicendo  dentro  di  sé  : \n—  Qualcuno  si  affaccerà.  — \n\nDifatti  si  affacciò  un  veccliio,  col  berretto  da \nuotte  in  capo,  il  quale  gridò  tutto  stizzito: \n—  Ohe  cosa  volete  a  quest'ora! \n\nHi  \"il     <x  mi  V»  /■ , \n\nTornò  a  casa  bagnato  come  un  pulcino.... \n\n—  Ohe  mi  fareste  il  piacere  di  darrai  un  po'  di \npane? \n—  Aspettatemi  costì  che  torno  subito,  —  rispose \nil  vecchino,  credendo  di  aver  da  fare  con  qualcuno \ndi  quei  ragazzacci  rompicolli  che  si  divertono  di \nnotte  a  sonare  i  cani  inanelli  delle  case,  per  mo- \nlestare  la  gente  per  bene,  che  se  la  dorme,  tran- \nquillamente. \nDopo  mezzo  minuto  la  finestra  si  riaprì,  e  la \nvoce  del  solito  vecchino  gridò  a  Finocchio  : \n—  Fatti  sotto  e  para  il  cappello.  — \nPinocchio  che  non  aveva  ancora  un  cappello, \nsi  avvicinò  e  sentì  pioversi  addosso  un'enorme \ncatinellata  d'acqua  che  lo  annaffiò  tutto,  dalla \ntesta  ai  piedi,  come  se  fosse  un  vaso  di  giranio \nappassito. \nTornò  a  casa  bagnato  come  un  pulcino  e  ri- \nfinito dalla  stanchezza  e  dalla  fame:  e  perchè \nnon  aveva  più  forza  di  reggersi  ritto,  si  pose  a \nsedere,  appoggiando  i  piedi  fradici  e  impillac- \ncherati sopra  un  caldano  pieno  di  brace  accesa. \nE  lì  si  addormentò;  e  nel  dormire  i  piedi  che \nerano  di  legno  gli  presero  fuoco,  e  adagio  adagio \ngli  si  carbonizzarono  e  diventarono  cenere. \nE  Pinocchio  seguitava  a  dormire  e  a  russare, \ncome  se  i  suoi  piedi  fossero  quelli  d'un  altro. \nFinalmente  sul  far  del  giorno  si  svegliò,  perchè \nqualcuno  aveva  bussato  alla  porta. \n—  Chi  è?  —  domandò  sbadigliando  e  stropic- \nciandosi gli  occhi. \n—  Sono  io  !  —  rispose  una  voce. \nQuella  voce  era  la  voce  di  Geppetto,`"
]

# === Funzione di generazione pulita ===
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        return_dict_in_generate=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Estrae solo il testo dopo "Risposta:"
    if "Risposta:" in decoded:
        return decoded.split("Risposta:")[-1].strip()
    else:
        return decoded.strip()

# === Prompt ===
def make_prompt(ocr_text):
    return f"Correggi: {ocr_text}\nRisposta:"

# === Comparazione ===
print("\n=== CONFRONTO MINERVA BASE vs FINE-TUNED ===\n")
for i, ocr in enumerate(ocr_examples):
    prompt = make_prompt(ocr)
    base_output = generate(model_base, prompt)
    fine_output = generate(model_finetuned, prompt)

    print(f"--- ESEMPIO {i+1} ---")
    print(f"OCR:         {ocr}")
    print(f"Base:        {base_output}")
    print(f"Fine-tuned:  {fine_output}")
    print()