import google.generativeai as genai

## This function sends a message to the Gemini AI model and returns the response
## LIMITATIONS: we have 500 RPD (Requests Per Day) for the free tier
def ask_gemini(message):
    genai.configure(api_key="AIzaSyAvfJXiEgC25NBZh2qkWT6LkbSgGDSPgrc")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(message)

    return response.text