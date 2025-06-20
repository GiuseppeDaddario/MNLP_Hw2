import json

def annotate_human_scores(FILE_NAME, correction_model):
    
    BASE_PATH = "datasets/eng/corrections/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)


    key = correction_model+"_correction"
    for i, item in enumerate(data):
        print("\n" + "=" * 50)
        print(f"Item {i+1}/{len(data)}")
        print("\033[91m- CORRETTO:\n\033[0m", item.get("corretto", ""))
        print(f"\033[91m- {correction_model} CORRECTION:\n\033[0m", item.get(key, ""))
        
        while True:
            try:
                score = int(input("Insert human score (1-5): "))
                if score < 1 or score > 5:
                    print(" must be from 1 to 5.")
                    continue
                item[f"human_{correction_model}_score"] = score
                break
            except ValueError:
                print(" Inserisci un numero valido.")

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n  \033[92mAnnotations complete and saved!\033[0m")

# ESEMPIO USO
# annotate_human_scores("tuo_file.json")

#to_annotate_path = "datasets/eng/corrections/finetuning_correction.json"
#annotate_human_scores(to_annotate_path)