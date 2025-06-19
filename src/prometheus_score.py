import json
import time

# IMPORTA IL CLIENT PROMETHEUS (sostituisci con il client che usi)
from prometheus_eval import PrometheusClient

# Inizializza il client (modifica l'endpoint e API key se serve)
client = PrometheusClient(api_key="LA_TUA_API_KEY")

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


def prometheus_ask_score(translation, expected, print_result=True):
    prompt = JUDGE_PROMPT.format(translation=translation, reference=expected)

    # Chiamata a Prometheus Eval (esempio generico)
    response = client.generate(prompt=prompt, max_tokens=20, temperature=0.0)
    
    score = response.strip()  # Pulisce spazi e newline

    if print_result:
        print("Score: " + score)

    return score


def prometheus_score(FILE_NAME):
    BASE_PATH = "datasets/eng/corrections/"
    FILE_PATH = BASE_PATH + FILE_NAME + ".json"

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data, start=1):
        translation = entry["llama4_correction"]
        reference = entry["corretto"]

        print(f"{i}/{len(data)} - Valutazione in corso...")

        try:
            score = prometheus_ask_score(translation, reference, print_result=False)
            entry["machine_score"] = int(score)
        except ValueError:
            entry["machine_score"] = score
        except Exception as e:
            print(f"Errore alla voce {i}: {e}")
            entry["machine_score"] = "ERROR"

        time.sleep(4.5)  # Rispetta eventuali limiti rate

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
