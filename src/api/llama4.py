
import requests

def ask_llama4(prompt):
    API_KEY = "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"  # Here is the API key
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"Errore: risposta API priva di 'choices': {res_json}"
    else:
        return f"Errore HTTP {response.status_code}: {response.text}"

