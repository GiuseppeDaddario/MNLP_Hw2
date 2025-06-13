import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# huggingface-cli login in order to use the model

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sapienzanlp/Minerva-350M-base-v1.0")
    model = AutoModelForCausalLM.from_pretrained("sapienzanlp/Minerva-350M-base-v1.0")
    return model, tokenizer

def ask_minerva(message):
    model_id = "sapienzanlp/Minerva-350M-base-v1.0"

    # Initialize the pipeline.
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Input text for the model.
    input_text = message

    # Compute the outputs.
    output = pipeline(
        input_text,
        max_new_tokens=128,
    )

    # Output:
    # [{'generated_text': "La capitale dell'Italia è la città di Roma, che si trova a [...]"}]

    return output[0]['generated_text']

if __name__ == "__main__":
    # Example usage
    response = ask_minerva("What is the capital of France?")
    print(response)  # Should print: "La capitale dell'Italia è Roma."
