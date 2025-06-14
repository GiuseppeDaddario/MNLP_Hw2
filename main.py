import torch

from src.utils import OcrCorrectionDataset, read_dataset, difference_score
from src.models  import t5Model
from src.api import ask_llama, ask_gemini

DIRTY_PATH_ENG = "datasets/eng/the_vampyre_ocr.json"
CLEAN_PATH_ENG = "datasets/eng/the_vampyre_clean.json"
DIRTY_PATH_ITA = "datasets/ita/original_ocr.json"
CLEAN_PATH_ITA = "datasets/ita/cleaned.json"

BASE_PROMPT = "I will send you a text done with ocr. You will correct it preserving the syntax and you will reply writing only the corrected text, without any other sentence.\n Example: 'This iss jusst a text and sonne err0rs 1n the phnase.'\n Your answer: 'This is just a text and some errors in the phrase.'\n\n"
JUDGE_PROMPT = ("Evaluate the quality of the output based on the following scale:\n"
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
                "coherent. It retained the original meaning as much as it could.\n"
                "Answer only with a number from 1 to 5, without any other sentence.\n\n")



def main_t5():

    # ==== 1. Inizializzazione ====
    model = t5Model(model_name="t5-base").to("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 2. Dataset ====
    dataset = OcrCorrectionDataset(DIRTY_PATH_ENG, CLEAN_PATH_ENG, model.tokenizer)

    # ==== 3. Training ====
    model.fit(train_dataset=dataset)

    example = "This iss jusst a text and sonne err0rs 1n the phnase."
    corrected = model(example)
    print("Corretto:", corrected[0])


def main():
    # reading datasets
    mandatory = read_dataset(DIRTY_PATH_ENG,5)
    correct = read_dataset(CLEAN_PATH_ENG,5)

    # asking the model to correct the text
    num_elem = 4
    prompt = f"{BASE_PROMPT} {mandatory[num_elem]}"
    output = ask_llama(prompt)
    print("api answer:", output)
    print("original  :", correct[num_elem])

    # Automaic score calculation based on the difference between the original and the output
    score = difference_score(correct[num_elem], output)
    print("Score:", score["score"])
    print("Missing words:", score["missing_words"])
    print("Added words:", score["added_words"])

    # LLM as a Judge
    eval_prompt = f"{JUDGE_PROMPT} Original: {correct[num_elem]} \n Output: {output}"
    judge_score = ask_gemini(eval_prompt)
    print("Judge score:", judge_score)


    return

if __name__ == "__main__":
    main()