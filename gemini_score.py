
import google.generativeai as genai


API_KEY = "AIzaSyD2SsbKDiwbxstu97xaC0HOG8Lhp6gw2DU"



# Configura la tua API key
genai.configure(api_key="AIzaSyD2SsbKDiwbxstu97xaC0HOG8Lhp6gw2DU")


#models = genai.list_models()
#for m in models:
#    print(m.name, "→", m.supported_generation_methods)

# Carica il modello Gemini-Pro
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Invia una richiesta

prompt = judge_prompt + "\n" + "[E]"

response = model.generate_content(prompt)

# Stampa la risposta
print(response.text)