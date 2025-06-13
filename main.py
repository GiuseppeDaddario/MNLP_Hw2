from src.dataset import OcrCorrectionDataset
from src.models.t5_model import t5Model
import torch

def main_t5():

    # ==== 1. Inizializzazione ====
    model = t5Model(model_name="t5-base").to("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 2. Dataset ====
    dirty_path = "datasets/eng/the_vampyre_ocr.json"
    clean_path = "datasets/eng/the_vampyre_clean.json"
    dataset = OcrCorrectionDataset(dirty_path, clean_path, model.tokenizer)

    # ==== 3. Training ====
    model.fit(train_dataset=dataset)

    example = "This iss jusst a text and sonne err0rs 1n the phnase."
    corrected = model(example)
    print("Corretto:", corrected[0])


def main():
    return

if __name__ == "__main__":
    main_t5()