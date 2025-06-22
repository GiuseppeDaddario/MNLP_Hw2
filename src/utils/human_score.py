import json
from rouge_score import rouge_scorer

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

#==== Human annotation ========
def annotate_human_scores(FILE_NAME, correction_model):
    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    #data = list(data.values())
    key = "correction"

    for i, item in enumerate(data):
        reference = item.get("gold", "")
        prediction = item.get(key, "")

        r1 = rouge_1(reference, prediction)
        r2 = rouge_2(reference, prediction)
        rl = rouge_l(reference, prediction)

        print("\n" + "=" * 50)
        print(f"Item {i+1}/{len(data)}")
        print("\033[91m- CORRETTO:\n\033[0m", reference)
        print(f"\033[91m- {correction_model} CORRECTION:\n\033[0m", prediction)
        print(f"ROUGE-1: {r1:.3f} | ROUGE-2: {r2:.3f} | ROUGE-L: {rl:.3f}")

        while True:
            try:
                score = int(input("Insert human score (1-5): "))
                if score < 1 or score > 5:
                    print(" must be from 1 to 5.")
                    continue
                item[f"human_score"] = score
                break
            except ValueError:
                print(" Inserisci un numero valido.")

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n  \033[92mAnnotations complete and saved!\033[0m")