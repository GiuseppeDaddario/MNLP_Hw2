
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re
from src.utils.logger import log

#==== LOADING MINERVA MODEL =====
def load_minerva_model(correction_model, finetuned=True, online=True):
    # Checking if running in Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    if online: #Just to be sure in case of local env on colab
        is_colab = True

    HF_BASE_MODEL_NAME = "sapienzanlp/Minerva-7B-instruct-v1.0"
    LOCAL_BASE_MODEL_PATH = "./src/models/minerva/cache/models--sapienzanlp--Minerva-7B-instruct-v1.0/snapshots/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
    FINETUNED_MODEL_PATH = f"./src/models/minerva/{correction_model}"

    # Model to upload
    if is_colab:
        BASE_MODEL_PATH = HF_BASE_MODEL_NAME
        local_files_only = False
        log("Running in Google Colab")
    else:
        BASE_MODEL_PATH = LOCAL_BASE_MODEL_PATH
        local_files_only = True
        log("Running in Local Environment")


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
    model_path = FINETUNED_MODEL_PATH if finetuned else BASE_MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    return model, tokenizer
#=======================


#==== MINERVA PROMPT MAKER =====
def make_prompt(correction_model, ocr_text):
    if correction_model == "minerva" or correction_model == "minerva_finetuned_llima":
        return (
             "You are an OCR correction system.\n"
             "Task: Fix spelling, spacing, and OCR errors in the given sentence ONLY if there are any.\n"
             "Rules:\n"
             "1. DO NOT explain.\n"
             "2. DO NOT change names, old spellings or historical terms.\n"
             "3. DO NOT add any text.\n"
             "4. If the sentence is already correct, repeat it exactly.\n"
             "Sentence: " + ocr_text + "\n"
             "Corrected:"
         )
    elif correction_model == "minerva_finetuned_post_ocr":
        return (
            "You are an OCR correction system. You MUST output the input exactly if it's already correct."
            "NEVER add, complete, or guess missing content. Only fix OCR errors."
            f"Input Sentence: {ocr_text}"
            "Corrected:"
        )
    return None
#=======================


#==== SINGLE GENERATION =====
@torch.inference_mode()
def ask_minerva(prompt, model, tokenizer):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Corrected:" in decoded:
        after_corrected = decoded.split("Corrected:", 1)[-1].strip()
        first_line = after_corrected.split("\n")[0].strip()
        return first_line, decoded

    return decoded.strip().split("\n")[0].strip(), decoded
#=======================


#==== TEXT PREPROCESSING =====
def split_into_sentences(text):
    # Divide with the point followed by a space or newline
    return [s.strip() for s in re.split(r'(?<=[.])\s+', text) if s.strip()]
#=======================


#==== PROCESSING DATASET =====
def process_ocr_file(correction_model, input_file, gold_file, output_file, model, tokenizer, force_indices=None, print_result=False):
    with open(input_file, "r", encoding="utf-8") as f_in:
        ocr_data = json.load(f_in)

    with open(gold_file, "r", encoding="utf-8") as f_gold:
        gold_data = json.load(f_gold)

    # Load output if exists, for skipping already processed keys
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_out:
            results = json.load(f_out)
    else:
        results = {}
    keys = list(ocr_data.keys())

    # Control for forced indices
    if force_indices is not None:
        keys = [keys[i] for i in force_indices if i < len(keys)]

    for i, key in enumerate(keys):
        # Skip if already processed and not forced
        if key in results and (force_indices is None or i not in force_indices):
            log(f"[{i+1}/{len(keys)}] Skipping already processed key '{key}'")
            continue

        ocr_text = ocr_data.get(key, "").strip()
        if not ocr_text:
            continue

        gold_text = gold_data.get(key, "").strip()
        sentences = split_into_sentences(ocr_text)
        corrected_sentences = []

        log(f"[{i+1}/{len(keys)}] Processing key '{key}'... ({len(sentences)} sentences)")
        
        for j, sentence in enumerate(sentences):
            if not sentence:
                continue
            prompt = make_prompt(correction_model,sentence)
            corrected, full_text = ask_minerva(prompt, model, tokenizer)
            corrected_sentences.append(corrected)

            #print(f"------------ DEBUG [{j+1}/{len(sentences)}] --------------")
            #print(f"{full_text}")
            #print("------------------------------------------")

        final_correction = " ".join(corrected_sentences)

        if print_result:
            log(f"OCR:        {ocr_text}")
            log(f"Gold:       {gold_text}")
            log(f"Correction: {final_correction}\n")

        results[key] = {
            "ocr": ocr_text,
            "gold": gold_text,
            "minerva_correction": final_correction
        }

        #==== SAVING ====
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)
        #================


#==== WRAPPER FUNCTION =====
def correct_with_minerva(file_name, correction_model="minerva", print_result=False, finetuned=False):

    input_path = f"datasets/{file_name}_ocr.json"
    gold_path = f"datasets/{file_name}_clean.json"
    output_path = f"datasets/corrections/{correction_model}/{file_name}.json"

    log("Loading Minerva model...")
    model, tokenizer = load_minerva_model(correction_model,finetuned=finetuned, online=True)

    log("Starting OCR correction...")
    process_ocr_file(correction_model, input_path, gold_path, output_path, model, tokenizer, force_indices=None, print_result=print_result)

    log(f"Corrections saved to:{output_path}")
#=============================