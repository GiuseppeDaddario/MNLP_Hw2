import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#==== PLOT ONE CONF MATRIX ====
def plot_confusion_matrix(FILE_NAME, correction_model, evaluation_model, normalize=False, cmap="Blues"):
    
    BASE_PATH = f"datasets/corrections/{correction_model}/"
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
#=====================


#==== PLOT TWO CONF MATRICES TOGETHER ====
def plot_confusion_matrices(FILE_NAME, correction_model, normalize=False, cmap="Blues"):
    BASE_PATH = f"datasets/corrections/{correction_model}/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    evaluation_model_1 = "gemini"
    evaluation_model_2 = "prometheus"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    def extract_scores(eval_model):
        human_scores = []
        machine_scores = []
        key = f"{eval_model}_score"

        for item in data:
            if "human_score" in item and key in item:
                human_scores.append(item["human_score"])
                machine_scores.append(item[key])
            else:
                print(f"Missing 'human_score' or '{key}' in item:", item)

        return human_scores, machine_scores

    # Extract scores for both models
    h1, m1 = extract_scores(evaluation_model_1)
    h2, m2 = extract_scores(evaluation_model_2)

    # Build confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, human_scores, machine_scores, eval_model in zip(
            axes, [h1, h2], [m1, m2], [evaluation_model_1, evaluation_model_2]
    ):
        labels = sorted(list(set(human_scores + machine_scores)))
        cm = confusion_matrix(human_scores, machine_scores, labels=labels, normalize='true' if normalize else None)

        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Model Score")
        ax.set_ylabel("Human Score")
        ax.set_title(f"{eval_model} vs Human")

    plt.suptitle(f"Confusion Matrices for '{correction_model}'", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./plots/confusion_matrices_{correction_model}.pdf",format='pdf', bbox_inches='tight',dpi=300)
    plt.show()
#=====================
