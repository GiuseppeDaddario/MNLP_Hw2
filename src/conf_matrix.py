import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(FILE_NAME, correction_model, evaluation_model, normalize=False, cmap="Blues"):
    
    BASE_PATH = "datasets/eng/corrections/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    human_scores = []
    machine_scores = []

    key = f"{correction_model}_{evaluation_model}_score"

    for item in data:
        if f"human_{correction_model}_score" in item and key in item:
            human_scores.append(item[f"human_{correction_model}_score"])
            machine_scores.append(item[key])
        else:
            print("Missing 'human_score' or 'machine_score' in item:", item)

    labels = sorted(list(set(human_scores + machine_scores)))
    cm = confusion_matrix(human_scores, machine_scores, labels=labels, normalize='true' if normalize else None)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Model Score")
    plt.ylabel("Human Score")
    plt.title(f"Evaluating {correction_model} translation: \n Human score vs {evaluation_model}")
    plt.tight_layout()
    plt.show()




