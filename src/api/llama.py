import transformers
import torch


def ask_llama(message):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Example usage
    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #     {"role": "user", "content": "Who are you?"},
    # ]

    outputs = pipeline(
        message,
        max_new_tokens=256,
    )

    return outputs[0]["generated_text"][-1]


if __name__ == "__main__":
    # Example usage
    response = ask_llama("What is the capital of France?")
    print(response)  # Should print: "The capital of France is Paris."
