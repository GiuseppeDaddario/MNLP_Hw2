import time
import json


################# DA COMPLETARE FUNZIONE PER QUERY PROMETHEUS ############
####
####
####
####

def prometheus_ask_score(translation, reference, print_result = True):
    
    response = "4"

    return response
####
####
####
####
####
####
################# DA COMPLETARE FUNZIONE PER QUERY PROMETHEUS!!!!!!!!! ############







###### FROM HERE IS OK ################

def prometheus_score(FILE_NAME, correction_model):
    

    BASE_PATH = "datasets/eng/corrections/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    # Carica il tuo JSON da file
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = f"{correction_model}_correction"
    key2 = f"{correction_model}_prometheus_score"



    print("|====================================")
    print(f"|--- Valutazione di {correction_model} con Prometheus ---|")
    for i, entry in enumerate(data, start=1):
        
        translation = entry[key]
        reference = entry["corretto"]

        print(f"{i}/{len(data)}")

        try:
            score = prometheus_ask_score(translation, reference, print_result=False)
            entry[key2] = int(score)
        except ValueError:
            entry[key2] = score  
        except Exception as e:
            print(f"Errore alla voce {i}: {e}")
            entry[key2] = "ERROR"

        time.sleep(4.5) 

    # Salva il risultato in un nuovo file
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("====================================")