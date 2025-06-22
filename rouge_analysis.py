import json
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# FUNZIONI ROUGE
def rouge_1(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(reference, prediction)['rouge1'].fmeasure

def rouge_2(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    return scorer.score(reference, prediction)['rouge2'].fmeasure

def rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)['rougeL'].fmeasure


def analyze_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    human_scores = []
    
    for item in data:
        ref = item.get("gold", "")
        pred = item.get("correction", "")
        human_score = item.get("human_score", None)
        
        # Calcolo ROUGE
        r1 = rouge_1(ref, pred)
        r2 = rouge_2(ref, pred)
        rl = rouge_l(ref, pred)
        
        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        rougel_scores.append(rl)
        
        if human_score is not None:
            human_scores.append(human_score)
        else:
            human_scores.append(np.nan)
    
    # Converti in numpy arrays per comodità
    rouge1_scores = np.array(rouge1_scores)
    rouge2_scores = np.array(rouge2_scores)
    rougel_scores = np.array(rougel_scores)
    human_scores = np.array(human_scores)
    
    print("=== STATISTICHE ROUGE ===")
    for name, scores in zip(["ROUGE-1", "ROUGE-2", "ROUGE-L"], [rouge1_scores, rouge2_scores, rougel_scores]):
        print(f"{name}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}")
    
    print("\n=== CORRELAZIONI (Spearman) tra ROUGE e punteggi umani ===")
    # Rimuovi nan per correlazioni
    mask = ~np.isnan(human_scores)
    for name, scores in zip(["ROUGE-1", "ROUGE-2", "ROUGE-L"], [rouge1_scores, rouge2_scores, rougel_scores]):
        corr, pval = spearmanr(scores[mask], human_scores[mask])
        print(f"{name}: Spearman r = {corr:.3f}, p-value = {pval:.3e}")
    
    # Boxplot distribuzione ROUGE
    plt.boxplot([rouge1_scores, rouge2_scores, rougel_scores], labels=["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    plt.title("Distribuzione punteggi ROUGE su dataset")
    plt.show()

# Esempio di uso (modifica il path al file json):
analyze_dataset("datasets/eng/corrections/t5/the_vampyre.json")
