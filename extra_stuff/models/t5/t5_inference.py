import torch
from transformers import pipeline
from spellchecker import SpellChecker
import re

def preprocess_text(text):
    return text.replace('1', 'i').replace('0', 'o').replace('4', 'a')

def spellcheck_text(text, spell):
    words = text.split()
    corrected_words = [spell.correction(w) if spell.correction(w) else w for w in words]
    return " ".join(corrected_words)

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]

if __name__ == "__main__":
    spell = SpellChecker(language='en')
    ocr_corrector = pipeline("text2text-generation", model="t5-base", device=0 if torch.cuda.is_available() else -1)

    ocr_input = "The gvardians hastened to protect Miss Aubrey; but wben they arrivcd, it was too 1 ate."

    sentences = split_into_sentences(ocr_input)
    corrected_sentences = []

    for sentence in sentences:
        preprocessed = preprocess_text(sentence)
        try:
            corrected = ocr_corrector(
                preprocessed,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )[0]['generated_text']
        except Exception as e:
            print(f"⚠️  Error during model inference: {e}")
            corrected = preprocessed

        final_sentence = spellcheck_text(corrected, spell)
        corrected_sentences.append(final_sentence)

    final_correction = " ".join(corrected_sentences)

    print(f"\n🔹 OCR Input:  {ocr_input}")
    print(f"✅ Correction: {final_correction}")