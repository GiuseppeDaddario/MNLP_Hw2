import json
from torch.utils.data import Dataset

class OcrCorrectionDataset(Dataset):
    def __init__(self, dirty_path, clean_path, tokenizer, max_length=128):
        with open(dirty_path, 'r', encoding='utf-8') as f:
            self.dirty = json.load(f)
        with open(clean_path, 'r', encoding='utf-8') as f:
            self.clean = json.load(f)

        self.keys = sorted(self.dirty.keys(), key=lambda x: int(x))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        dirty_text = "fix: " + self.dirty[key]
        clean_text = self.clean[key]

        inputs = self.tokenizer(dirty_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(clean_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignora il padding nel loss

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels
        }


def read_dataset(file_path, elements=None,start_idx=None):
    if elements is None:
        if "eng" in file_path:
            elements = 24
        elif "ita" in file_path:
            elements = 7
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Creiamo una lista con solo i primi 'elements' elementi
    if start_idx is not None:
        keys = sorted(data.keys(), key=lambda x: int(x))[elements:]
    else:
        keys = sorted(data.keys(), key=lambda x: int(x))[:elements]
    return [data[k] for k in keys]


def difference_score(original, corrected):
    original_words = set(original.split())
    corrected_words = set(corrected.split())

    # Parole corrette che coincidono
    correct_words = original_words.intersection(corrected_words)

    # Parole mancanti e aggiunte
    missing_words = original_words - corrected_words  # Parole presenti nell'originale ma mancanti nel corretto
    added_words = corrected_words - original_words    # Parole presenti nel corretto ma non nell'originale

    # Calcola la percentuale di correzione
    if len(original_words) == 0:
        score = 0.0
    else:
        score = len(correct_words) / len(original_words) * 100

    # Restituisce punteggio e dettagli delle differenze
    return {
        "score": score,
        "missing_words": missing_words,
        "added_words": added_words
    }

import json

def build_ocr_dataset(ocr_path, clean_path, output_path, start_idx=8):
    ocr_data = read_dataset(ocr_path, start_idx=start_idx)
    clean_data = read_dataset(clean_path, start_idx=start_idx)
    assert len(ocr_data) == len(clean_data), "OCR e testi corretti non hanno la stessa lunghezza!"
    dataset = [{"ocr": o, "corretto": c} for o, c in zip(ocr_data, clean_data)]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Dataset salvato in: {output_path} ({len(dataset)} esempi)")


if __name__ == "__main__":
    ocr_file = "../../datasets/ita/original_ocr.json"        # Cambia se necessario
    clean_file = "../../datasets/ita/cleaned.json"    # Cambia se necessario
    output_file = "../../datasets/ita/finetuning.json"

    build_ocr_dataset(ocr_file, clean_file, output_file)
