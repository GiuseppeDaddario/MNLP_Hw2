
import google.generativeai as genai
import json
import time


FILE_PATH = r"datasets\eng\corrections\finetuning_correction.json"



def gemini_ask_score(translation,expected,print_result=True):

    # Configura la tua API key
    genai.configure(api_key="AIzaSyD2SsbKDiwbxstu97xaC0HOG8Lhp6gw2DU")
    model = genai.GenerativeModel("gemini-1.5-flash-latest")



    JUDGE_PROMPT = ("You are evaluating the quality of a translated sentence by comparing it to a perfect reference translation.\n\n"
                    "Use the following scale to assign a score from 1 to 5:\n"
                    "1. Completely unacceptable output: the output has no "
                    "pertinence with the original meaning, the generated sentence is "
                    "either gibberish or something that makes no sense.\n"
                    "2. Severe semantic errors, omissions or substantial add ons on the "
                    "original sentence. The errors are of semantic and syntactic nature. "
                    "It’s still something no human would ever write.\n"
                    "3. Partially wrong output, the output is lackluster, it contains "
                    "errors, but are mostly minor errors, like typos, or small semantic "
                    "errors.\n"
                    "4. Good output. The output is mostly right, substantially "
                    "faithful to the original text, but the style does not perfectly match the "
                    "original sentence, still fluent and comprehensible, and could "
                    "semantically acceptable.\n "
                    "5. Perfect output. The output is accurate, fluent, complete and "
                    "coherent. It retained the original meaning as much as it could.\n\n"
                    "Compare the following:\n"
                    "Translation: \"{translation}\"\n"
                    "Reference: \"{reference}\"\n\n"
                    "JUST YOUR SCORE (1 to 5):")


    prompt = JUDGE_PROMPT.format(translation=translation, reference=expected)

    
    response = model.generate_content(prompt)
    score = response.text.strip() ##Removes spaces and new lines

    # Stampa la risposta
    if print_result:
        print("Score: " + score)
    
    return score



def gemini_score(FILE_NAME, correction_model):

    BASE_PATH = "datasets/eng/corrections/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"
    # Carica il tuo JSON da file
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = f"{correction_model}_correction"
    key2 = f"{correction_model}_gemini_score"

    print("|========================================")
    print(f"|--- Valutazione di {correction_model} con Gemini ---|")
    for i, entry in enumerate(data, start=1):
        translation = entry[key]
        reference = entry["corretto"]

        print(f"{i}/{len(data)}")

        try:
            score = gemini_ask_score(translation, reference, print_result=False)
            entry[key2] = int(score)
        except ValueError:
            entry[key2] = score  # fallback se Gemini restituisce qualcosa di strano
        except Exception as e:
            print(f"Errore alla voce {i}: {e}")
            entry[key2] = "ERROR"

        time.sleep(4.5)  # Per rispettare il limite del free tier (15/min)

    # Salva il risultato in un nuovo file
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("========================================")





