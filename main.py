import torch

from src.utils import OcrCorrectionDataset
from src.models  import t5Model
from src.api import ask_llama

DIRTY_PATH = "datasets/eng/the_vampyre_ocr.json"
CLEAN_PATH = "datasets/eng/the_vampyre_clean.json"

def main_t5():

    # ==== 1. Inizializzazione ====
    model = t5Model(model_name="t5-base").to("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 2. Dataset ====
    dataset = OcrCorrectionDataset(DIRTY_PATH, CLEAN_PATH, model.tokenizer)

    # ==== 3. Training ====
    model.fit(train_dataset=dataset)

    example = "This iss jusst a text and sonne err0rs 1n the phnase."
    corrected = model(example)
    print("Corretto:", corrected[0])


def main():
    base_prompt = "Correggi rispettando la punteggiatura, senza aggiungere altro: "
    prompt = f"{base_prompt} The universa1 belief js, that a person sucked by a vampyre becomes a vampyre himself, arid sucks in his turn."
    output = ask_llama(prompt)

    return

if __name__ == "__main__":
    main_t5()