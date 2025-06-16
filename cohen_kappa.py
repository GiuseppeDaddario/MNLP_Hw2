from sklearn.metrics import cohen_kappa_score
import json


def kappa_correlation(file_path="", print_results=True):
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    human_scores = []
    machine_scores = []
    
    for item in data:
        if "human_score" in item and "machine_score" in item:
            human_scores.append(item["human_score"])
            machine_scores.append(item["machine_score"])


    kappa = cohen_kappa_score(human_scores, machine_scores)
    
    if print_results:
        print("\n")
        print("|------------------------------------------")
        print("| Remember that 0 < C-K < 1: \n| -- 0 means no agreement\n| -- 1 means complete agreement")
        print("|------------------------------------------")
        print(f"| Cohen's Kappa: {kappa:.3f}")
        print("|------------------------------------------")
        print("\n")
    return kappa



