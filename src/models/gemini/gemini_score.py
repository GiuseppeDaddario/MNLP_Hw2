import os
import json
import time
import re
from datetime import datetime
import google.generativeai as genai

# === LOGGING ===
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === PROMPT STATICO ===
JUDGE_PROMPT = (
    "Evaluate the quality of the [GENERATED] text in comparison to the [EXPECTED] text. "
    "Use the following scale:\n\n"
    "1 = Completely unacceptable. The output is severely incomplete or entirely incorrect.\n"
    "2 = Major issues. The output is complete but significantly inaccurate or misleading.\n"
    "3 = Some errors. Mostly correct, but with noticeable mistakes or altered meaning.\n"
    "4 = Minor issues. Largely accurate, with only small grammatical or factual deviations.\n"
    "5 = Perfect. No errors; completely faithful and correct.\n\n"
    "Output only the score after the [NUMERIC SCORE] tag.\n"
    "Format: [NUMERIC SCORE] <number>\n"
    "Add a brief explanation of the score.\n"
)

# === PARSER DELLO SCORE ===
def extract_numeric_score(text):
    match = re.search(r"\[NUMERIC SCORE\]\s*[:\-]?\s*(\d)", text)
    return int(match.group(1)) if match else f"[ERRORE: {text.strip()}]"

# === INIZIALIZZA GEMINI ===
def init_gemini():
    API_KEY = os.environ.get("GEMINI_KEY", "")
    if not API_KEY:
        raise EnvironmentError("GEMINI_API_KEY non impostata nelle variabili d'ambiente.")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    return model

# === VALUTA UN ESEMPIO ===
def gemini_ask_score(original, reference, correction, model):
    prompt = (
        f"{JUDGE_PROMPT}\n\n"
        f"[OCR]\n{original}\n\n"
        f"[EXPECTED]\n{reference}\n\n"
        f"[GENERATED]\n{correction}\n\n"
        f"[NUMERIC SCORE]"
    )
    response = model.generate_content(prompt)
    decoded = response.text.strip()

    #print("\n--- OUTPUT GREZZO DI GEMINI ---")
    #print(decoded)
    #print("--- FINE OUTPUT ---\n")

    return extract_numeric_score(decoded)

# === FUNZIONE COMPLETA DI SCORING ===
def gemini_score(FILE_NAME, correction_model):
    model = init_gemini()

    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    FILE_PATH = os.path.join(BASE_PATH, f"{FILE_NAME}.json")
    groupname = "C0rr3tt0r1_4ut0m4t1c1"
    evaluation_model = "gemini"
    JUDGE_PATH = f"outputs/{correction_model}/{groupname}-hw2_ocr-{evaluation_model}.json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    key_input = "correction"
    key_output = "gemini_score"

    log("|====================================")
    log("| \033[34mEvaluating with Gemini...\033[0m")

    judge_output_data = {}  # Dizionario, chiave: indice come stringa

    for i, entry in enumerate(data):
        try:
            original = entry["ocr"]
            reference = entry["gold"]
            correction = entry[key_input]

            # Qui modifica gemini_ask_score per restituire anche output raw,
            # oppure ricostruisci output con prompt e risposta se vuoi
            prompt = (
                f"{JUDGE_PROMPT}\n\n"
                f"[OCR]\n{original}\n\n"
                f"[EXPECTED]\n{reference}\n\n"
                f"[GENERATED]\n{correction}\n\n"
                f"[NUMERIC SCORE]"
            )
            response = model.generate_content(prompt)
            decoded = response.text.strip()
            score = extract_numeric_score(decoded)

            entry[key_output] = int(score) if isinstance(score, int) else score

            judge_output_data[str(i)] = {
                "output": decoded,
                "score": entry[key_output]
            }

        except Exception as e:
            log(f"Errore alla voce {i+1}: {e}")
            entry[key_output] = "ERROR"
            judge_output_data[str(i)] = {
                "output": f"ERROR: {e}",
                "score": "ERROR"
            }

        log(f"{i+1}/{len(data)} - Score: {entry[key_output]}")
        time.sleep(4.5)  # Rispetta rate limit

    # Salva i dati con gli score aggiornati
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Salva il file di valutazione dettagliato (JUDGE_PATH)
    os.makedirs(os.path.dirname(JUDGE_PATH), exist_ok=True)
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(judge_output_data, f, ensure_ascii=False, indent=2)

    log(f"Evaluation file saved: {JUDGE_PATH}")
    log("|====================================")