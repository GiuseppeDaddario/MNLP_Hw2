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
