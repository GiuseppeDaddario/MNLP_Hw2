
import google.generativeai as genai
import json
import time

INPUT_FILE = "datasets/eng/finetuning.json"
OUTPUT_FILE = "datasets/eng/finetuning_splitted.json"

# Configura l'API key di Gemini
genai.configure(api_key="AIzaSyD2SsbKDiwbxstu97xaC0HOG8Lhp6gw2DU")
model = genai.GenerativeModel("gemini-1.5-flash-latest")

SPLIT_PROMPT = (
    "Split the following text into smaller segments of **no more than 20 words each**.\n"
    "You must:\n"
    "- Keep **all punctuation, spelling, and structure intact**.\n"
    "- **Do NOT add or change** anything.\n"
    "- Just break the text naturally at punctuation or word boundaries to stay under 20 words per chunk.\n"
    "- Return the result as a **JSON array of strings**.\n\n"
    "Text:\n\"{text}\"\n\n"
    "Result (JSON list):"
)

def ask_gemini_split(text, print_result=False):
    prompt = SPLIT_PROMPT.format(text=text)
    response = model.generate_content(prompt)
    try:
        chunks = json.loads(response.text.strip())
        if print_result:
            print(chunks)
        return chunks
    except Exception as e:
        print("Parsing error:", e)
        print("Gemini output:", response.text)
        return ["ERROR"]

def split_dataset():
    # Carica il dataset originale
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    print("|========================================")
    print(f"| \033[93mSplitting dataset with Gemini...\033[0m")
    for i, entry in enumerate(data, 1):
        print(f"| {i}/{len(data)}", end="\r", flush=True)

        new_entry = dict(entry)  # copia per sicurezza

        try:
            split_ocr = ask_gemini_split(entry["ocr"], print_result=False)
            split_correct = ask_gemini_split(entry["corretto"], print_result=False)

            new_entry["ocr_split"] = split_ocr
            new_entry["correct_split"] = split_correct
        except Exception as e:
            print(f"\nErrore alla voce {i}: {e}")
            new_entry["ocr_split"] = ["ERROR"]
            new_entry["correct_split"] = ["ERROR"]

        new_data.append(new_entry)
        time.sleep(4.5)  # rispetto rate limit del free tier

    print("\n| Salvataggio del nuovo file...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("|✅ File salvato in:", OUTPUT_FILE)
    print("|========================================")

if __name__ == "__main__":
    split_dataset()





