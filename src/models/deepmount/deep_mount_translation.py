from transformers import pipeline
from spellchecker import SpellChecker
import json
import re
from datetime import datetime

def init():
    spell = SpellChecker(language='en')
    ocr_corrector = pipeline("text2text-generation", model="DeepMount00/OCR_corrector")
    return spell, ocr_corrector

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def preprocess_text(text):
    return text.replace('1', 'i').replace('0', 'o').replace('4', 'a')

def spellcheck_text(text,spell):
    words = text.split()
    corrected_words = [spell.correction(w) if spell.correction(w) else w for w in words]
    return " ".join(corrected_words)

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]

def translate_with_deep_mount(FILE_NAME, print_result=True):
    spell, ocr_corrector = init()
    print("\n")
    print("|========================================")
    print("| \033[93mTranslating with smaLLM ...\033[0m")

    input_path = f"datasets/eng/{FILE_NAME}_ocr.json"
    gold_path = f"datasets/eng/{FILE_NAME}_clean.json"
    output_path = f"datasets/eng/corrections/deep_mount/{FILE_NAME}.json"

    with open(input_path, "r", encoding="utf-8") as f_in:
        ocr_data = json.load(f_in)

    with open(gold_path, "r", encoding="utf-8") as f_gold:
        gold_data = json.load(f_gold)

    results = {}

    for i, key in enumerate(ocr_data.keys()):
        ocr_text = ocr_data[key].strip()
        gold_text = gold_data.get(key, "").strip()

        if not ocr_text:
            continue

        sentences = split_into_sentences(ocr_text)
        corrected_sentences = []

        log(f"[{i+1}/{len(ocr_data)}] Processing key '{key}' ({len(sentences)} sentences)")

        for j, sentence in enumerate(sentences):
            preprocessed = preprocess_text(sentence)

            try:
                corrected = ocr_corrector(preprocessed)[0]['generated_text']
            except Exception as e:
                print(f"⚠️  Error during model inference: {e}")
                corrected = preprocessed  # fallback

            final_sentence = spellcheck_text(corrected,spell)
            corrected_sentences.append(final_sentence)

            print(f"  ↪️ [{j+1}/{len(sentences)}] {final_sentence}")

        final_correction = " ".join(corrected_sentences)

        log(f"📝 OCR:        {ocr_text}")
        log(f"🤖 Correction: {final_correction}")
        log(f"✅ Gold:       {gold_text}\n")

        results[key] = {
            "ocr": ocr_text,
            "gold": gold_text,
            "smaLLM_correction": final_correction
        }

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    if print_result:
        print(f"\nRisultati salvati in: {output_path}")

    print("|========================================")
    print("\n")