## LORE

# huggingface-cli login in order to use the model


from huggingface_hub import login

# Inserisci qui il tuo token personale
#hf_token = "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" #istituzionale
hf_token = "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"  #privato

def HF_Login():
    login(token=hf_token)

## LORE