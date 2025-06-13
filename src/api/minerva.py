import transformers
import torch

# huggingface-cli login in order to use the model


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
