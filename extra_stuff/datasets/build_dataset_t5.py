import json
from pathlib import Path

def split_into_chunks(text, max_words=10):
    """Dividi il testo in blocchi da massimo `max_words` parole."""
    words = text.strip().split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def process_dataset(file_paths, max_words=10):
    all_chunks = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            ocr_chunks = split_into_chunks(entry.get("ocr", ""), max_words=max_words)
            correct_chunks = split_into_chunks(entry.get("corretto", ""), max_words=max_words)

            # Allinea le coppie (anche se uno è più lungo, tronca alla lunghezza minima)
            for ocr, corr in zip(ocr_chunks, correct_chunks):
                all_chunks.append({
                    "ocr": ocr,
                    "corretto": corr
                })

    return all_chunks

if __name__ == "__main__":
    input_files = ["human_data.json", "finetuning.json"]
    output_file = "finetuning_t5.json"

    final_dataset = process_dataset(input_files, max_words=10)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(final_dataset, f_out, ensure_ascii=False, indent=2)

    print(f"✅ Dataset finale salvato in: {output_file} ({len(final_dataset)} elementi)")