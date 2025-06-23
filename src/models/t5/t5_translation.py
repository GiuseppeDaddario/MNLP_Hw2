from transformers import pipeline, AutoTokenizer
from spellchecker import SpellChecker
import spacy
import json
import re
from src.utils.logger import log

#==== CONFIGURATION ====
nlp = spacy.load("en_core_web_sm")
MAX_TOKENS = 100
MIN_CHUNK_WORDS = 3  # minimo parole per chunk valido
#=======================

#==== INIT ====
def init():
    spell = SpellChecker(language='en')
    ocr_corrector = pipeline("text2text-generation", model="yelpfeast/byt5-base-english-ocr-correction")
    tokenizer = AutoTokenizer.from_pretrained("yelpfeast/byt5-base-english-ocr-correction")
    return spell, ocr_corrector, tokenizer
#===============

#==== PREPROCESSING ====
def preprocess_text(text): #Common substitutions and cleaning
    text = text.replace('1', 'i').replace('0', 'o').replace('4', 'a')
    text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # caratteri non ascii
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text): # Split text into sentences using spacy
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# Chunk sentence into smaller parts based on token count (for tokenizer limits)
def chunk_sentence_by_tokens(sentence, tokenizer, max_tokens=MAX_TOKENS):
    words = sentence.split()
    chunks = []
    current_chunk = []

    for word in words:
        test_chunk = current_chunk + [word]
        if len(test_chunk) < MIN_CHUNK_WORDS:
            current_chunk.append(word)
            continue

        token_count = len(tokenizer(" ".join(test_chunk), return_tensors="pt")["input_ids"][0])
        if token_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def capitalize_first_letter(text):
    if not text:
        return text
    return text[0].upper() + text[1:]
#=============================


#==== SPELLCHECKING ====
def spellcheck_word(word, spell):
    if not re.match(r"^[a-zA-Z'-]+$", word):
        return word
    correction = spell.correction(word)
    if correction is None:
        return word
    return correction

def spellcheck_text(text, spell):
    words = text.split()
    corrected_words = [spellcheck_word(w, spell) for w in words]
    return " ".join(corrected_words)

#========================


#==== PROCESS THE DATASET ====
def correct_with_t5(FILE_NAME, print_result=True):
    spell, ocr_corrector, tokenizer = init()
    print("\n|========================================")
    print("| \033[34mTranslating with t5 ...\033[0m")

    input_path = f"datasets/eng/{FILE_NAME}_ocr.json"
    gold_path = f"datasets/eng/{FILE_NAME}_clean.json"
    output_path = f"datasets/eng/corrections/t5/{FILE_NAME}.json"

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

        for sent_i, sent in enumerate(sentences, 1):
            sub_chunks = chunk_sentence_by_tokens(sent, tokenizer, max_tokens=MAX_TOKENS)
            corrected_sub_chunks = []

            log(f"  Sentence {sent_i}/{len(sentences)}: {len(sub_chunks)} chunks")

            for chunk_i, chunk in enumerate(sub_chunks, 1):
                preprocessed = preprocess_text(chunk)
                try:
                    out = ocr_corrector(f"fix: {preprocessed}")[0]['generated_text']
                    if out.lower().startswith("fix:"):
                        corrected = out[len("fix:"):].strip()
                    else:
                        corrected = out.strip()

                    # fallback if output is too short or empty
                    if len(corrected) < 5 or corrected in ["...", "fix...", ""]:
                        log(f"    Chunk {chunk_i}/{len(sub_chunks)}: Model output too short, using preprocessed text")
                        corrected = preprocessed
                except Exception as e:
                    log(f"    Chunk {chunk_i}/{len(sub_chunks)}: Error: {e}")
                    corrected = preprocessed

                spellchecked = spellcheck_text(corrected, spell)
                corrected_sub_chunks.append(spellchecked)

            corrected_sentence = " ".join(corrected_sub_chunks)
            corrected_sentence = capitalize_first_letter(corrected_sentence)
            corrected_sentences.append(corrected_sentence)

        # Building the final correction
        final_correction = " ".join(corrected_sentences)

        if print_result:
            log(f"OCR:        {ocr_text}")
            log(f"Gold:       {gold_text}")
            log(f"Correction: {final_correction}\n")

        results[key] = {
            "ocr": ocr_text,
            "gold": gold_text,
            "correction": final_correction
        }

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print("|========================================\n")