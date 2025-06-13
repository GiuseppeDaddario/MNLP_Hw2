
import requests

def groq_chat(api_key, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
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

if __name__ == "__main__":
    API_KEY = "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"  # Here is the API key
    prompt = "Correggi: 'il c4ne è suL c0modino '" ## Insert here your prompt

    output = groq_chat(API_KEY, prompt)
    print("Risposta API:")
    print(output)
