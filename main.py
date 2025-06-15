import torch

from src.utils import OcrCorrectionDataset, read_dataset, difference_score
from src.models  import t5Model
from src.api import ask_llama, ask_gemini, absolute_grading

DIRTY_PATH_ENG = "/leonardo/home/userexternal/gdaddari/MNLP_Hw2/datasets/eng/the_vampyre_ocr.json"
CLEAN_PATH_ENG = "/leonardo/home/userexternal/gdaddari/MNLP_Hw2/datasets/eng/the_vampyre_clean.json"
DIRTY_PATH_ITA = "/leonardo/home/userexternal/gdaddari/MNLP_Hw2/datasets/ita/original_ocr.json"
CLEAN_PATH_ITA = "/leonardo/home/userexternal/gdaddari/MNLP_Hw2/datasets/ita/cleaned.json"

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

INSTRUCTION = "Correct the sentence removing OCR errors and typos, preserving the original meaning and syntax. Do not add any additional text or explanation."


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
    print("Reading datasets...")
    mandatory = read_dataset(DIRTY_PATH_ENG,5)
    correct = read_dataset(CLEAN_PATH_ENG,5)

    # asking the model to correct the text
    print("Asking model ...")
    num_elem = 4
    prompt = f"{BASE_PROMPT} {mandatory[num_elem]}"
    output = ask_llama(prompt)
    print("Llama ans :", output)
    print("original  :", correct[num_elem])

    # Automaic score calculation based on the difference between the original and the output
    score = difference_score(correct[num_elem], output)
    print("Score:", score["score"])
    print("Missing words:", score["missing_words"])
    print("Added words:", score["added_words"])

    # LLM as a Judge
    print("Evaluating with LLMs...")
    eval_prompt = f"{JUDGE_PROMPT} Original: {correct[num_elem]} \n Output: {output}"
    gemini_score = ask_gemini(eval_prompt)
    print("Gemini score:", gemini_score)
    if torch.cuda.is_available():
        prometheus_score = absolute_grading(instruction=INSTRUCTION, response=output, reference_answer=correct[num_elem], score_rubric=JUDGE_PROMPT)
        print("Prometheus score:", prometheus_score["score"])

    return

def main_leonardo():
    # reading datasets
    output = "In the London Journal, of March, 1732, is a curious, and, of course, credible account of a particular case of vampirism, which is stated to have occurred at Madreyga, in Hungary. It appears, that upon an examination of the commander-in-chief and magistrates of the place, they positively and unanimously affirmed, that, about five years before, a certain Heyduke, named Arnold Paul, had been heard to say, that, at Cassovia, on the frontiers of the Turkish Servia, he had been tormented by a vampyre, but had found a way to rid himself of the evil, by eating some of the earth out of the vampyre's grave, and rubbing himself with his blood. This precaution, however, did not prevent him from becoming a vampyre himself; for, about twenty or thirty days after his death and burial, many persons complained of having been tormented by him, and a deposition was made, that four persons had been deprived of life by his attacks. To prevent further mischief, the inhabitants having consulted their Hadagni, took up the body, and found it (as is supposed to be usual in cases of vampirism) fresh, and entirely free from corruption, and emitting at the mouth, nose, and ears, pure and florid blood. Proof having been thus obtained, they resorted to the accustomed remedy. A stake was driven entirely through the heart and body of Arnold Paul, at which he is reported to have cried out as dreadfully as if he had been alive. This done, they cut off his head, burned his body, and threw the ashes into his grave. The same measures were adopted with the corpses of those persons who had previously died from vampirism, lest they should, in their turn, become agents upon others who survived them."
    correct = "In the London Journal, of March, 1732, is a curious, and, of course, credible account of a particular case of vampyrism, which is stated to have occurred at Madreyga, in Hungary. It appears, that upon an examination of the commander-in-chief and magistrates of the place, they positively and unanimously affirmed, that, about five years before, a certain Heyduke, named Arnold Paul, had been heard to say, that, at Cassovia, on the frontiers of the Turkish Servia, he had been tormented by a vampyre, but had found a way to rid himself of the evil, by eating some of the earth out of the vampyre's grave, and rubbing himself with his blood. This precaution, however, did not prevent him from becoming a vampyre himself; for, about twenty or thirty days after his death and burial, many persons complained of having been tormented by him, and a deposition was made, that four persons had been deprived of life by his attacks. To prevent further mischief, the inhabitants having consulted their Hadagni, took up the body, and found it (as is supposed to be usual in cases of vampyrism) fresh, and entirely free from corruption, and emitting at the mouth, nose, and ears, pure and florid blood. Proof having been thus obtained, they resorted to the accustomed remedy. A stake was driven entirely through the heart and body of Arnold Paul, at which he is reported to have cried out as dreadfully as if he had been alive. This done, they cut off his head, burned his body, and threw the ashes into his grave. The same measures were adopted with the corses of those persons who had previously died from vampyrism, lest they should, in their turn, become agents upon others who survived them."
    
    # LLM as a Judge
    print("Evaluating with Prometheus...")
    if torch.cuda.is_available():
        prometheus_score = absolute_grading(instruction=INSTRUCTION, response=output, reference_answer=correct, score_rubric=JUDGE_PROMPT)
        print("Prometheus score:", prometheus_score["score"])

    return

if __name__ == "__main__":
    main_leonardo()