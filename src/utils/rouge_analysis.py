import json
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#==== ROUGE FUNCTIONS ========
def rouge_1(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(reference, prediction)['rouge1'].fmeasure

def rouge_2(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    return scorer.score(reference, prediction)['rouge2'].fmeasure

def rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)['rougeL'].fmeasure

def build_rouges(FILE_NAME, correction_model):
    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    INPUT_PATH = BASE_PATH + FILE_NAME + ".json"
    OUTPUT_PATH = BASE_PATH + FILE_NAME + "_rouges.json"

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #data = list(data.values())
    key = "correction"

    rouges = {}

    for i, item in enumerate(data):
        reference = item.get("gold", "")
        prediction = item.get(key, "")

        rouge1 = rouge_1(reference, prediction)
        rouge2 = rouge_2(reference, prediction)
        rougel = rouge_l(reference, prediction)

        rouges[str(i)] = {
            "rouge-1": round(rouge1, 4),
            "rouge-2": round(rouge2, 4),
            "rouge-l": round(rougel, 4)
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rouges, f, ensure_ascii=False, indent=2)

    print(f"ROUGE scores saved to: {OUTPUT_PATH}")


def analyze_dataset(FILE_NAME, correction_model):
    # Carica i dati del dataset (gold, correction, human_score)
    data_file_path = f"../../datasets/eng/corrections/{correction_model}/{FILE_NAME}.json"
    rouge_file_path = f"../../datasets/eng/corrections/{correction_model}/{FILE_NAME}_rouges.json"

    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Carica i ROUGE dal file esterno
    with open(rouge_file_path, "r", encoding="utf-8") as f:
        rouge_data = json.load(f)

    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    human_scores = []

    for i, item in enumerate(data):
        idx = str(i)  # perché il file ROUGE ha chiavi stringa
        rouge_scores = rouge_data.get(idx)

        if rouge_scores:
            rouge1_scores.append(rouge_scores.get("rouge-1", np.nan))
            rouge2_scores.append(rouge_scores.get("rouge-2", np.nan))
            rougel_scores.append(rouge_scores.get("rouge-l", np.nan))
        else:
            print(f"[Warning] Nessun ROUGE trovato per indice {i}")
            rouge1_scores.append(np.nan)
            rouge2_scores.append(np.nan)
            rougel_scores.append(np.nan)

        # human_score può essere mancante
        human_scores.append(item.get("human_score", np.nan))

    # Converti in numpy arrays
    rouge1_scores = np.array(rouge1_scores)
    rouge2_scores = np.array(rouge2_scores)
    rougel_scores = np.array(rougel_scores)
    human_scores = np.array(human_scores)

    print("=== STATISTICHE ROUGE ===")
    for name, scores in zip(["ROUGE-1", "ROUGE-2", "ROUGE-L"], [rouge1_scores, rouge2_scores, rougel_scores]):
        print(f"{name}: mean={np.nanmean(scores):.3f}, std={np.nanstd(scores):.3f}, min={np.nanmin(scores):.3f}, max={np.nanmax(scores):.3f}")

    print("\n=== CORRELAZIONI (Spearman) tra ROUGE e punteggi umani ===")
    # Maschera per filtrare valori validi
    mask = ~np.isnan(human_scores)
    for name, scores in zip(["ROUGE-1", "ROUGE-2", "ROUGE-L"], [rouge1_scores, rouge2_scores, rougel_scores]):
        valid_mask = mask & ~np.isnan(scores)
        corr, pval = spearmanr(scores[valid_mask], human_scores[valid_mask])
        print(f"{name}: Spearman r = {corr:.3f}, p-value = {pval:.3e}")

    # Boxplot distribuzione ROUGE
    plt.boxplot([rouge1_scores, rouge2_scores, rougel_scores], labels=["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    plt.title(f"ROUGE scores distribution on the dataset {FILE_NAME}, {correction_model}")
    plt.ylabel("Score")
    plt.show()

# Esempio di uso (modifica il path al file json):
if __name__ == "__main__":
    analyze_dataset("the_vampyre","minerva")
