from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from spellchecker import SpellChecker
import re
import string
import json
from tqdm import tqdm

def init():
    model_path = "lorebenucci/DeepReworkedMount"
    #model_path = "DeepMount00/OCR_corrector"
    #model_path = "yelpfeast/byt5-base-english-ocr-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    ocr_corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    spell = SpellChecker(language='en')
    return model_path,tokenizer,model,ocr_corrector,spell

def correct_spelling(text, spell):
    words = text.split()
    corrected_words = []
    for word in words:
        # Corregge solo se parola non è riconosciuta
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    return " ".join(corrected_words)

def clean_text(text):
    text = text.strip()

    # Rimuovi caratteri strani all’inizio
    text = re.sub(r'^[^A-Za-z]+', '', text)

    # Spazi multipli
    text = re.sub(r'\s{2,}', ' ', text)

    # Capitalizza solo la prima lettera della frase
    if text:
        text = text[0].upper() + text[1:]

    # Aggiungi punto finale se manca
    if text and text[-1] not in string.punctuation:
        text += "."

    return text

def ocr_fix(ocr_corrector,text: str, spell) -> str:
    input_text = f"{text}"
    model_output = ocr_corrector(input_text, max_new_tokens=256)[0]['generated_text']
    corrected = correct_spelling(model_output, spell)

    cleaned = clean_text(corrected)
    return cleaned

def correct_with_deep_mount(FILE_NAME, print_result=False):
    model_path, tokenizer, model, ocr_corrector, spell = init()
    file = FILE_NAME
    datapath = "datasets/eng/"

    input_path = datapath + file + "_ocr.json"
    gold_path = datapath + file + "_clean.json"
    output_path = datapath + "corrections/deep_mount/" + file + ".json"


    # Carica file OCR e GOLD
    with open(input_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # Lista finale dei record
    dataset_corretto = []

    for key in tqdm(ocr_data.keys(), desc="Correggendo dataset"):
        ocr_text = ocr_data[key]
        gold_text = gold_data.get(key, "")

        # Correggi il testo OCR con il tuo modello+spellchecker
        correction = ocr_fix(ocr_corrector,ocr_text, spell)

        # Aggiungi il record alla lista
        dataset_corretto.append({
            "ocr": ocr_text,
            "gold": gold_text,
            "correction": correction
        })

        if print_result:
            print(f"\nOCR:        {ocr_text}")
            print(f"Gold:       {gold_text}")
            print(f"Correction: {correction}\n")

    # Salva su file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_corretto, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved in: {output_path}")
