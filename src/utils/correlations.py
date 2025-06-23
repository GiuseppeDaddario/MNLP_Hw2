from sklearn.metrics import cohen_kappa_score
import json

#==== KAPPA CORRELATION ===============
def kappa_correlation(FILE_NAME, correction_model, evaluation_model, print_results=True):
    
    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    human_scores = []
    machine_scores = []
    
    key = f"{evaluation_model}_score"

    for item in data:
        if f"human_score" in item and key in item:
            human_scores.append(item[f"human_score"])
            machine_scores.append(item[key])
        else:
            print("Missing 'human_score' or 'machine_score' in item:", item)

    kappa = cohen_kappa_score(human_scores, machine_scores)
    
    if print_results:

        print("\n")
        print("|====================================")
        print(f"| \033[34mKAPPA CORRELATION:\033[0m\n| translator: {correction_model}\n| judge: {evaluation_model}")
        print("|------------------------------------")
        print("| Remember that -1 < C-K < 1: \n| -- 0 means casual agreement\n| -- 1 means complete agreement\n| -- -1 means complete disagreement")
        print("|------------------------------------")
        print(f"| Cohen's Kappa: {kappa:.3f}")
        print("|====================================")
        print("\n")

    return kappa
#=======================================


#==== ACCURACY CORRELATION =============
def accuracy_correlation(FILE_NAME, correction_model, evaluation_model, print_results=True):
    
    BASE_PATH = f"datasets/eng/corrections/{correction_model}/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    human_scores = []
    machine_scores = []
    
    key = f"{evaluation_model}_score"
    for item in data:
        if f"human_score" in item and key in item:
            human_scores.append(item[f"human_score"])
            machine_scores.append(item[key])
        else:
            print("Missing 'human_score' in item:", item)
    
    matches = [1 if h == m else 0 for h, m in zip(human_scores, machine_scores)]
    accuracy = sum(matches) / len(matches) if matches else 0
    
    if print_results:

        print("\n")
        print("|====================================")
        print(f"| \033[34mACCURACY CORRELATION:\033[0m\n| translator: {correction_model}\n| judge: {evaluation_model}")
        print("|------------------------------------")
        print("| Defined as: \n| Matching human-machine scores/total")
        print("|------------------------------------")
        print(f"| Accuracy: {accuracy:.3f}")
        print("|====================================")
        
        print("\n")

    return accuracy
#=========================================



